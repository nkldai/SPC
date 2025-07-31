import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import fnmatch

from matplotlib.ticker import MaxNLocator, FuncFormatter, ScalarFormatter
from matplotlib.dates import DateFormatter
from datetime import timedelta


# plt.rcParams['backend'] = 'tkagg'
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Times new roman'
# plt.rcParams['mathtext.rm'] = 'serif'
# plt.rcParams['mathtext.it'] = 'serif:italic'
# plt.rcParams['mathtext.bf'] = 'serif:bold'
# plt.rcParams['mathtext.fontset'] = 'custom'


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

base_path = "C:/data/EEMS"
machine_ids = ["FEMS101_01", "FEMS101_02", "FEMS101_03", "FEMS101_04"]

""""""""""""""""""""""""
def process_and_save_parquet(): # csv --> parquet
    patterns = [
        "SPC팩 회전자개조_*_with_header.csv",
        "log_250716_Modification.csv"
    ]
    for root, dirs, files in os.walk(base_path):
        for pattern in patterns:
            for filename in fnmatch.filter(files, pattern):
                csv_path = os.path.join(root, filename)

                df = pd.read_csv(csv_path)
                numeric_cols = df.select_dtypes(include=['number']).columns
                df[numeric_cols] = df[numeric_cols].astype(np.float32)

                original_name = os.path.splitext(filename)[0]
                new_name = f"SPC-unfiltered-composed-{original_name}.parquet"
                parquet_path = os.path.join(root, new_name)

                df.to_parquet(parquet_path, engine='pyarrow')
                print(f"[Unfiltered] Saved: {parquet_path}")
# process_and_save_parquet()

""""""""""""""""""""""""
def process_and_save_filtered_parquet(): # filtered
    pattern = "SPC-unfiltered-composed-*.parquet"
    keep_columns = ['collect_time', 'machine_code', 'Load_Total_Power_Consumption']

    for root, dirs, files in os.walk(base_path):
        for filename in fnmatch.filter(files, pattern):
            parquet_path = os.path.join(root, filename)
            df = pd.read_parquet(parquet_path)

            numeric_cols = df[keep_columns].select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].astype(np.float32)

            df = df[keep_columns]

            new_name = filename.replace("unfiltered", "filtered")
            new_path = os.path.join(root, new_name)

            df.to_parquet(new_path, engine='pyarrow')
            print(f"[Filtered] Saved: {new_path}")
# process_and_save_filtered_parquet()

""""""""""""""""""""""""
def combine_and_split_by_machine(): # decomposed
    pattern = "SPC-filtered-composed-*.parquet"
    parquet_files = []

    for root, dirs, files in os.walk(base_path):
        for filename in fnmatch.filter(files, pattern):
            parquet_files.append(os.path.join(root, filename))

    if not parquet_files:
        print("⚠️ No matching parquet files found.")
        return

    df_list = [pd.read_parquet(f) for f in parquet_files]
    full_df = pd.concat(df_list, ignore_index=True)

    full_df['collect_time'] = pd.to_datetime(full_df['collect_time'], errors='coerce')
    full_df = full_df.reset_index(drop=True)

    for machine_id in machine_ids:
        df_machine = full_df[full_df['machine_code'] == machine_id]

        save_path = os.path.join(base_path, f"SPC-filtered-decomposed-{machine_id}.parquet")
        df_machine.to_parquet(save_path, engine='pyarrow', index=False)
        print(f"✅ Saved {save_path} with {len(df_machine)} rows")
# combine_and_split_by_machine()

""""""""""""""""""""""""
def split_chunk_into_24h_subchunks_aligned(chunk_df, ref_time): # 24시간 주기 맞추기
    subchunks = []

    chunk_start = chunk_df['collect_time'].min()
    chunk_end   = chunk_df['collect_time'].max()

    aligned_start = pd.Timestamp.combine(chunk_start.date(), ref_time)

    step = pd.Timedelta(hours=24)
    current_start = aligned_start

    while current_start + step <= chunk_end:
        current_end = current_start + step - pd.Timedelta(seconds=1)
        mask = (chunk_df['collect_time'] >= current_start) & (chunk_df['collect_time'] <= current_end)
        sub_df = chunk_df.loc[mask]

        if not sub_df.empty:
            subchunks.append(sub_df)

        current_start += step

    return subchunks

""""""""""""""""""""""""
def get_long_chunks(): # 1시간 downtime + 24시간 chunk
    downtime_threshold = pd.Timedelta(hours=1)
    min_duration = pd.Timedelta(hours=24)

    results = {}

    for m_id in machine_ids:
        file_path = os.path.join(base_path, f"SPC-filtered-decomposed-{m_id}.parquet")
        df = pd.read_parquet(file_path)

        df = df.sort_values('collect_time').reset_index(drop=True)

        global_start_time = df['collect_time'].iloc[0].time()

        gap = df['collect_time'].diff()
        chunk_id = (gap >= downtime_threshold).cumsum()
        df['chunk_id'] = chunk_id

        long_chunks = []
        chunk_counter = 1

        for chunk_id, chunk_df in df.groupby('chunk_id'):
            duration = chunk_df['collect_time'].iloc[-1] - chunk_df['collect_time'].iloc[0]
            if duration >= min_duration:
                subchunks = split_chunk_into_24h_subchunks_aligned(chunk_df, global_start_time)

                for subchunk in subchunks:
                    if not subchunk.empty:
                        filename = f"SPC-filtered-decomposed-{m_id}-split-{chunk_counter}.parquet"
                        save_path = os.path.join(base_path, filename)
                        subchunk.to_parquet(save_path, engine='pyarrow', index=False)
                        print(f"✅ Saved: {save_path} ({len(subchunk)} rows)")
                        chunk_counter += 1

                long_chunks.extend(subchunks)

        results[m_id] = long_chunks

        print(f"\n📌 {m_id}: {len(long_chunks)} chunks saved (each ≈ 24h)")
        for i, chunk in enumerate(long_chunks):
            start = chunk['collect_time'].iloc[0]
            end = chunk['collect_time'].iloc[-1]
            print(f"   └─ Chunk {i + 1}: {start} → {end} (duration: {end - start}, rows: {len(chunk)})")

    return results
# get_long_chunks()


def plot_chunks_all():  # 1시간 downtime + 24시간 청크 별 시각화
    all_chunks = get_long_chunks()

    n_rows, n_cols = 5, 6
    total_plots = n_rows * n_cols

    for machine_id in machine_ids:
        chunks = all_chunks[machine_id]

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
        print(f"\n {machine_id} - 시작0 Load_Total_Power_Consumption")

        for idx in range(total_plots):
            row = idx // n_cols
            col = idx % n_cols
            ax = axs[row, col]

            if idx < len(chunks):
                df = chunks[idx].copy()

                start_y = df['Load_Total_Power_Consumption'].iloc[0]
                df['y_plot'] = df['Load_Total_Power_Consumption'] - start_y

                y_start = df['y_plot'].iloc[0]
                y_end = df['y_plot'].iloc[-1]
                y_min = df['y_plot'].min()
                y_max = df['y_plot'].max()
                duration = df['collect_time'].iloc[-1] - df['collect_time'].iloc[0]

                print(f" └─ Chunk {idx + 1:>2}: Start={y_start:.2f}, End={y_end:.2f}, Min={y_min:.2f}, Max={y_max:.2f}, Duration={duration}")

                ax.scatter(df['collect_time'], df['y_plot'], color='tab:blue', linewidth=2.0)
                ax.set_title(f"{machine_id} - Chunk {idx + 1}", fontsize=14)
                ax.tick_params(axis='x', labelsize=14, rotation=45)
                ax.tick_params(axis='y', labelsize=14)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.grid(True)
            else:
                ax.axis("off")


        fig.suptitle(f"{machine_id} - Power per Chunk", fontsize=18)
        plt.tight_layout()
        plt.show()
# plot_chunks_all()

""""""""""""""""""""""""
def get_long_chunks_debug_plot():  # diff 평균이 0.01 이상인 청크
    chunks_by_machine = get_long_chunks()
    final_results = {}
    min_diff_threshold = 0.01

    for m_id in machine_ids:
        long_chunks = chunks_by_machine[m_id]

        filtered_chunks = []
        print(f"\n✅ {m_id}: Checking {len(long_chunks)} chunks...")

        for i, chunk in enumerate(long_chunks):
            y = chunk['Load_Total_Power_Consumption']
            mean_diff = y.diff().abs().mean()

            if mean_diff < min_diff_threshold:
                start = chunk['collect_time'].iloc[0]
                end = chunk['collect_time'].iloc[-1]
                print(f"⛔ Chunk {i+1} skipped (mean diff {mean_diff:.6f} < {min_diff_threshold}): {start} → {end}")
                continue

            filtered_chunks.append(chunk)

        final_results[m_id] = filtered_chunks

        print(f"\n📊 {m_id}: {len(filtered_chunks)} valid chunks (each ≈ 24h)")
        for i, chunk in enumerate(filtered_chunks):
            start = chunk['collect_time'].iloc[0]
            end = chunk['collect_time'].iloc[-1]
            y = chunk['Load_Total_Power_Consumption']
            y_shifted = y - y.iloc[0]
            max_shifted = y_shifted.max()
            print(f"   └─ Chunk {i+1}: {start} → {end} (rows: {len(chunk)}, max_shifted: {max_shifted:.2f})")

        # 📈 시각화
        n_rows, n_cols = 4, 6
        total_plots = n_rows * n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5))

        for idx in range(total_plots):
            row = idx // n_cols
            col = idx % n_cols
            ax = axs[row, col]

            if idx < len(filtered_chunks):
                df = filtered_chunks[idx].copy()

                start_y = df['Load_Total_Power_Consumption'].iloc[0]
                y_shifted = df['Load_Total_Power_Consumption'] - start_y

                ax.scatter(df['collect_time'], y_shifted, color='tab:blue', linewidth=2.0)
                ax.set_title(f"Chunk {idx + 1}\n{df['collect_time'].iloc[0].strftime('%m-%d %H:%M')}", fontsize=13)
                ax.set_xlabel("Time", fontsize=14)
                ax.set_ylabel("Power (Δ)", fontsize=14)
                ax.tick_params(axis='x', labelsize=14, rotation=45)
                ax.tick_params(axis='y', labelsize=14)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.grid(True)
            else:
                ax.axis("off")

        fig.suptitle(f'{m_id} - Shifted 24h Chunks (Start = 0)', fontsize=18)
        plt.tight_layout()
        plt.show()

    return final_results
# get_long_chunks_debug_plot()


def analyze_chunk_power_change_rate(): # [0] [-1] 하나에 두개 다 그려서 비교 시각화
    all_chunks = get_long_chunks_debug_plot()

    for m_id, chunks in all_chunks.items():
        chunk0 = chunks[0]
        chunk_last = chunks[-1]

        start0 = chunk0['collect_time'].iloc[0]
        start_last = chunk_last['collect_time'].iloc[0]

        start_val0 = chunk0['Load_Total_Power_Consumption'].iloc[0]
        start_val_last = chunk_last['Load_Total_Power_Consumption'].iloc[0]

        def relative_hours(times, ref_time):
            delta = times - ref_time
            return delta.dt.total_seconds() / 3600

        x0_rel = relative_hours(chunk0['collect_time'], start0)
        x_last_rel = relative_hours(chunk_last['collect_time'], start_last)

        y0_shifted = chunk0['Load_Total_Power_Consumption'] - start_val0
        y_last_shifted = chunk_last['Load_Total_Power_Consumption'] - start_val_last

        def time_formatter(x, pos=None):
            new_time = start0 + timedelta(hours=x)
            return new_time.strftime("%H:%M")

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.scatter(x0_rel, y0_shifted, label='Chunk [0]', color='tab:blue', s=5)
        ax.scatter(x_last_rel, y_last_shifted, label='Chunk [-1]', color='tab:orange', s=5)

        ax.set_xlabel('collect_time', fontsize=14)
        ax.set_ylabel('Load_Total_Power_Consumption', fontsize=14)
        ax.set_title(f"{m_id} - Chunk [0] vs Chunk [-1]", fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))

        plt.tight_layout()
        plt.show()
# analyze_chunk_power_change_rate()

""""""""""""""""""""""""
def compare_chunk_slope_first_last():  # 청크[0] [-1]의 0 아닌 부분의 평균 기울기 + 변화량/변화율 (0 포함)
    all_chunks = get_long_chunks_debug_plot()

    def compute_average_slope(chunk):
        t = chunk['collect_time']
        y = chunk['Load_Total_Power_Consumption'] - chunk['Load_Total_Power_Consumption'].iloc[0]
        dt_hours = (t - t.iloc[0]).dt.total_seconds() / 3600

        mask = y != 0

        dy = y[mask].diff()
        dx = dt_hours[mask].diff()
        slope = dy / dx
        return slope.mean()

    def compute_change_and_rate(chunk):
        t = chunk['collect_time']
        y = chunk['Load_Total_Power_Consumption']
        dt_hours = (t.iloc[-1] - t.iloc[0]).total_seconds() / 3600
        dy = y.iloc[-1] - y.iloc[0]
        rate = dy / dt_hours if dt_hours != 0 else float('nan')
        return dy, rate

    for m_id, chunks in all_chunks.items():
        chunk0 = chunks[3]
        chunk_last = chunks[-1]

        slope_0 = compute_average_slope(chunk0)
        slope_last = compute_average_slope(chunk_last)

        dy0, rate0 = compute_change_and_rate(chunk0)
        dy_last, rate_last = compute_change_and_rate(chunk_last)

        print(f"{m_id}")
        print(f"   • Chunk [0] 전체 변화량:           {dy0:.2f} kW")
        print(f"   • Chunk [0] 전체 변화율:           {rate0:.4f} kW/시간")
        print(f"   • Chunk [0] 0 제외 변화율:   {slope_0:.4f} kW/시간\n")

        print(f"   • Chunk [-1] 전체 변화량:          {dy_last:.2f} kW")
        print(f"   • Chunk [-1] 전체 변화율:          {rate_last:.4f} kW/시간")
        print(f"   • Chunk [-1] 0 제외 변화율: {slope_last:.4f} kW/시간")
# compare_chunk_slope_first_last()

""""""""""""""""""""""""
def merge_chunks_by_flat_hold_strict(chunk, max_flat_hold_hours=0.1): # flat 구간이 0.1시간 이상 유지 되는 구간 기준 청크 분할, flat 구간 전까지 서브 청크 생성
    t = chunk['collect_time'].reset_index(drop=True)
    y = chunk['Load_Total_Power_Consumption'].reset_index(drop=True)
    n = len(y)

    subchunks = []
    start_idx = 0
    i = 0

    while i < n - 1:
        delta_y = y[i + 1] - y[i]

        if delta_y < 0.1:
            hold_start = i
            while i < n - 1 and (y[i + 1] - y[i]) < 0.1:
                i += 1
            hold_end = i
            hold_duration = (t[hold_end] - t[hold_start]).total_seconds() / 3600

            if hold_duration >= max_flat_hold_hours:
                if start_idx < hold_start:
                    subchunks.append(chunk.iloc[start_idx:hold_start].copy())
                start_idx = hold_end + 1
                i = hold_end + 1
                continue
        i += 1

    if start_idx < n:
        subchunks.append(chunk.iloc[start_idx:].copy())

    return subchunks

""""""""""""""""""""""""
def print_merged_chunks_with_avg_slope_by_flat_hold_strict(max_flat_hold_hours=0.2): # 0 vs -1 서브청크 평균변화율 + 가중 평균
    all_chunks = get_long_chunks_debug_plot()
    chunk_index_map = {
        f'FEMS101_0{i}': [0, -1] for i in range(1, 5)  # 앞, 뒤 청크
    }

    for m_id, chunks in all_chunks.items():
        print(f"\n📊 Machine: {m_id}")

        target_indices = chunk_index_map.get(m_id, [0, -1])
        for idx in target_indices:
            actual_idx = idx if idx >= 0 else len(chunks) + idx

            if actual_idx < 0 or actual_idx >= len(chunks):
                print(f"⚠️ 청크 인덱스 {idx}는 {m_id}의 범위를 벗어남.")
                continue

            chunk = chunks[actual_idx]
            merged_subchunks = merge_chunks_by_flat_hold_strict(
                chunk, max_flat_hold_hours=max_flat_hold_hours
            )

            print(f"\n🔹 Chunk [{idx}] - ΔY≒0이고 ΔT≥{max_flat_hold_hours}h 기준 병합 결과:")

            total_delta_y = 0.0
            total_delta_t = 0.0
            count_valid = 0

            for i, subchunk in enumerate(merged_subchunks):
                t0 = subchunk['collect_time'].iloc[0]
                t1 = subchunk['collect_time'].iloc[-1]
                y0 = subchunk['Load_Total_Power_Consumption'].iloc[0]
                y1 = subchunk['Load_Total_Power_Consumption'].iloc[-1]
                delta_t = (t1 - t0).total_seconds() / 3600
                delta_y = y1 - y0
                slope = delta_y / delta_t if delta_t > 0 else float('nan')

                if abs(delta_y) < 1e-6 or abs(slope) < 1e-6:
                    continue

                total_delta_y += delta_y
                total_delta_t += delta_t
                count_valid += 1

                print(f"   • Subchunk {i}: {t0} → {t1} | ΔY = {delta_y:.1f}, ΔT = {delta_t:.2f}h → slope = {slope:.2f} kW/h")

            if count_valid > 0 and total_delta_t > 0:
                weighted_avg_slope = total_delta_y / total_delta_t
                print(f"\n   ✅ 서브청크 {count_valid}개 → 가중 평균 변화율 = {weighted_avg_slope:.2f} kW/h")
# print_merged_chunks_with_avg_slope_by_flat_hold_strict()


def plot_all_machines_with_subchunk_starts(): # 서브 청크 시각화 + 시작 시간 표시
    all_chunks = get_long_chunks_debug_plot()
    max_flat_hold_hours = 0.2
    chunk_indices = [0, -1]
    default_colors = ['blue', 'red']

    if len(chunk_indices) > len(default_colors):
        raise ValueError("chunk_indices 수보다 색상 수가 적습니다. default_colors를 늘려주세요.")

    for machine_id in machine_ids:
        chunks = all_chunks.get(machine_id, [])
        if not chunks:
            print(f"No chunks found for {machine_id}")
            continue

        for idx, chunk_index in enumerate(chunk_indices):
            actual_idx = chunk_index if chunk_index >= 0 else len(chunks) + chunk_index
            if actual_idx < 0 or actual_idx >= len(chunks):
                print(f"Chunk index {chunk_index} out of range for {machine_id}")
                continue

            chunk = chunks[actual_idx]
            subchunks = merge_chunks_by_flat_hold_strict(chunk, max_flat_hold_hours=max_flat_hold_hours)

            t = chunk['collect_time'].reset_index(drop=True)
            y_raw = chunk['Load_Total_Power_Consumption'].reset_index(drop=True)
            y_min = y_raw.min()
            y = y_raw - y_min

            color_scatter = default_colors[idx]
            color_line = 'darkred'

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.scatter(t, y, color=color_scatter, s=5)

            for subchunk in subchunks:
                sub_t0 = subchunk['collect_time'].iloc[0]
                sub_y0 = subchunk['Load_Total_Power_Consumption'].iloc[0] - y_min
                label = sub_t0.strftime("%H:%M")

                ax.plot([sub_t0, sub_t0], [sub_y0, 0], color=color_line, linestyle='--', linewidth=1.2)
                ax.text(sub_t0, 0, label, fontsize=9, rotation=45,
                        ha='center', va='top', color=color_line,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

            ax.set_xlabel('Time')
            ax.set_ylabel('Load_Total_Power_Consumption (offset)')
            ax.set_title(f"{machine_id} - Chunk {chunk_index} with Subchunk Start Times")
            ax.grid(True)
            ax.tick_params(axis='x', rotation=45, labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))

            plt.tight_layout()
            plt.show()
# plot_all_machines_with_subchunk_starts()


def print_merged_chunks_with_avg_slope_by_flat_hold_strict(max_flat_hold_hours=0.1, common_index=0, special_index=3): # dataframe을 만들기
    all_chunks = get_long_chunks_debug_plot()

    for m_id, chunks in all_chunks.items():
        print(f"\n📊 Machine: {m_id}")

        if m_id.endswith(('01', '02', '03', '04')):
            indices = [common_index, -1]
        elif m_id.endswith('05'):
            indices = [special_index, -1]
        else:
            indices = [0, -1]

        for idx in indices:
            if len(chunks) <= abs(idx):
                print(f"⚠️ Chunk[{idx}] 없음")
                continue

            actual_idx = idx if idx >= 0 else len(chunks) + idx
            chunk = chunks[actual_idx]
            merged_subchunks = merge_chunks_by_flat_hold_strict(chunk, max_flat_hold_hours=max_flat_hold_hours)

            print(f"\n🔹 Chunk [{idx}] - ΔY=0이고 ΔT≥{max_flat_hold_hours}h 기준으로 나눈 병합 청크 + 평균 변화율:")

            rows = []
            for i, subchunk in enumerate(merged_subchunks):
                t0 = subchunk['collect_time'].iloc[0]
                t1 = subchunk['collect_time'].iloc[-1]
                y0 = subchunk['Load_Total_Power_Consumption'].iloc[0]
                y1 = subchunk['Load_Total_Power_Consumption'].iloc[-1]
                delta_t = (t1 - t0).total_seconds() / 3600
                delta_y = y1 - y0
                slope = delta_y / delta_t if delta_t > 0 else float('nan')

                if abs(delta_y) < 1e-6 or abs(slope) < 1e-6:
                    continue

                rows.append({
                    'Subchunk': i,
                    'Start Time': t0,
                    'End Time': t1,
                    'ΔY': round(delta_y, 1),
                    'ΔT (h)': round(delta_t, 2),
                    'Slope (kW/h)': round(slope, 2)
                })

            df = pd.DataFrame(rows)
            print(df.to_string(index=False))
# print_merged_chunks_with_avg_slope_by_flat_hold_strict()

""""""""""""""""""""""""
def find_max_common_variation_interval(min_overlap_days=1): # 최소 변화율이 최대인 구간 찾기
    chunks_by_machine = get_long_chunks_debug_plot()

    date_map = {}
    for m_id in machine_ids:
        date_map[m_id] = [chunk['collect_time'].iloc[0].date() for chunk in chunks_by_machine.get(m_id, [])]

    common_dates = sorted(set.intersection(*[set(d) for d in date_map.values()]))
    if len(common_dates) < 2:
        print("⚠️ 공통 구간이 2개 미만입니다.")
        return None

    best_start, best_end = None, None
    best_min_rate_among_machines = -1
    interval_count = 0

    for i in range(len(common_dates)):
        for j in range(i + min_overlap_days, len(common_dates)):
            d_start = common_dates[i]
            d_end = common_dates[j]
            interval_count += 1

            min_rate_in_this_interval = float('inf')
            all_valid = True
            print(f"\n🧪 Checking interval: {d_start} → {d_end}")

            for m_id in machine_ids:
                chunks = chunks_by_machine[m_id]
                chunk_dict = {chunk['collect_time'].iloc[0].date(): chunk for chunk in chunks}

                if d_start not in chunk_dict or d_end not in chunk_dict:
                    print(f"⛔ {m_id} missing chunks for {d_start} or {d_end}")
                    all_valid = False
                    break

                chunk_start = chunk_dict[d_start]
                chunk_end = chunk_dict[d_end]

                start_val = chunk_start['Load_Total_Power_Consumption'].iloc[0]
                end_val = chunk_end['Load_Total_Power_Consumption'].iloc[-1]
                diff = abs(end_val - start_val)

                duration_hours = (chunk_end['collect_time'].iloc[-1] - chunk_start['collect_time'].iloc[0]).total_seconds() / 3600
                if duration_hours == 0:
                    print(f"⛔ {m_id} duration is zero hours!")
                    all_valid = False
                    break

                rate = diff / duration_hours
                min_rate_in_this_interval = min(min_rate_in_this_interval, rate)

                print(f"   └─ {m_id}: Δ = {diff:.2f}, Rate = {rate:.2f} per hour")

            if all_valid and min_rate_in_this_interval > best_min_rate_among_machines:
                best_min_rate_among_machines = min_rate_in_this_interval
                best_start, best_end = d_start, d_end
                print(f"✅ New best interval: {best_start} → {best_end} (min rate = {best_min_rate_among_machines:.2f}/h)")

    print(f"\n📊 총 비교한 구간 수: {interval_count}개")
    if best_start and best_end:
        print(f"🏁 최종 선택된 구간: {best_start} → {best_end} (min rate = {best_min_rate_among_machines:.2f}/h)")
        return best_start, best_end, chunks_by_machine
    else:
        print("❌ 유효한 구간을 찾지 못했습니다.")
        return None
# find_max_common_variation_interval()


def visualize_min_rate_motor(min_overlap_days=1): # 최소 변화율이 최대인 구간의 시작과 끝 부분 시각화
    best_start, best_end, chunks_by_machine = find_max_common_variation_interval(min_overlap_days)
    if not best_start or not best_end:
        print("No valid interval found.")
        return

    min_rate = float('inf')
    min_motor = None
    selected_chunks = None

    for m_id in machine_ids:
        chunks = chunks_by_machine[m_id]
        chunk_dict = {chunk['collect_time'].iloc[0].date(): chunk for chunk in chunks}

        if best_start not in chunk_dict or best_end not in chunk_dict:
            continue

        chunk_start = chunk_dict[best_start]
        chunk_end = chunk_dict[best_end]

        y0 = chunk_start['Load_Total_Power_Consumption'].iloc[0]
        y1 = chunk_end['Load_Total_Power_Consumption'].iloc[-1]
        diff = abs(y1 - y0)

        duration_hours = (chunk_end['collect_time'].iloc[-1] - chunk_start['collect_time'].iloc[0]).total_seconds() / 3600
        if duration_hours == 0:
            continue

        rate = diff / duration_hours
        if rate < min_rate:
            min_rate = rate
            min_motor = m_id
            selected_chunks = (chunk_start, chunk_end)

    if selected_chunks is None:
        print("No valid motor data for selected interval.")
        return

    chunk_start, chunk_end = selected_chunks

    def relative_hours(df):
        base_time = df['collect_time'].iloc[0]
        return (df['collect_time'] - base_time).dt.total_seconds() / 3600

    x_start = relative_hours(chunk_start)
    x_end = relative_hours(chunk_end)

    y_start = chunk_start['Load_Total_Power_Consumption'] - chunk_start['Load_Total_Power_Consumption'].iloc[0]
    y_end = chunk_end['Load_Total_Power_Consumption'] - chunk_end['Load_Total_Power_Consumption'].iloc[0]

    def time_formatter(x, pos=None):
        base_time = chunk_start['collect_time'].iloc[0]
        new_time = base_time + timedelta(hours=x)
        return new_time.strftime("%H:%M")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(x_start, y_start, s=6, color='tab:blue', label=f"{best_start}")
    ax.scatter(x_end, y_end, s=6, color='tab:orange', label=f"{best_end}")

    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Power", fontsize=14)
    ax.set_title(f"{min_motor} | Min Rate Interval\n{best_start} → {best_end}", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=12)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
# visualize_min_rate_motor()