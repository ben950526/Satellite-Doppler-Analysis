import streamlit as st
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite, utc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import uniform_filter1d
import tempfile

st.set_page_config(page_title="衛星星間通訊分析工具", layout="wide")
st.title("🛰️ 兩顆衛星星間通訊分析工具")
st.markdown("**上傳兩個 TLE 檔案，即可自動分析**")

# 側邊欄
with st.sidebar:
    st.header("分析參數")
    step_hours = st.slider("取樣間隔 (小時)", 1, 24, 6)
    comm_threshold = st.slider("可通訊距離門檻 (km)", 500, 5000, 2000, step=100)
    smooth_window = st.slider("平滑程度", 3, 15, 7)

# 上傳檔案
col1, col2 = st.columns(2)
with col1:
    tle1 = st.file_uploader("衛星1 TLE 檔案", type=["txt"])
with col2:
    tle2 = st.file_uploader("衛星2 TLE 檔案", type=["txt"])

if tle1 and tle2:
    if st.button("🚀 開始分析", type="primary", use_container_width=True):
        with st.spinner("計算中，請稍候..."):
            try:
                # 儲存檔案
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f1:
                    f1.write(tle1.getvalue())
                    path1 = f1.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f2:
                    f2.write(tle2.getvalue())
                    path2 = f2.name

                ts = load.timescale()

                # 載入 TLE
                def load_all_tles(filename):
                    tles = []
                    with open(filename, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f if line.strip()]
                    i = 0
                    while i < len(lines) - 1:
                        line1 = lines[i]
                        line2 = lines[i+1]
                        if line1.startswith('1 ') and line2.startswith('2 '):
                            try:
                                sat = EarthSatellite(line1, line2, "SAT", ts)
                                tles.append({
                                    'epoch': sat.epoch.utc_datetime(),
                                    'line1': line1,
                                    'line2': line2
                                })
                            except:
                                pass
                        i += 2
                    tles.sort(key=lambda x: x['epoch'])
                    return tles

                tle_list1 = load_all_tles(path1)
                tle_list2 = load_all_tles(path2)

                # 選擇最適合 TLE 的函數
                def get_satellite_at_time(tle_list, target_dt):
                    if not tle_list:
                        raise ValueError("TLE 列表為空")
                    best = min(tle_list, key=lambda x: abs((x['epoch'] - target_dt).total_seconds()))
                    return EarthSatellite(best['line1'], best['line2'], "SAT", ts)

                # 計算
                start_date = datetime(2023, 11, 11, tzinfo=utc)
                end_date = datetime(2024, 5, 11, tzinfo=utc)
                step_seconds = step_hours * 3600

                all_times = []
                all_distances = []
                all_radial_vel = []
                all_doppler = []

                current = start_date
                progress_bar = st.progress(0)

                while current < end_date:
                    batch_end = min(current + timedelta(days=30), end_date)
                    num_steps = int((batch_end - current).total_seconds() / step_seconds) + 1
                    dt_list = [current + timedelta(seconds=i*step_seconds) for i in range(num_steps)]
                    times_batch = ts.utc(dt_list)

                    for t, dt in zip(times_batch, dt_list):
                        sat1 = get_satellite_at_time(tle_list1, dt)
                        sat2 = get_satellite_at_time(tle_list2, dt)

                        pos1 = sat1.at(t).position.km
                        vel1 = sat1.at(t).velocity.km_per_s
                        pos2 = sat2.at(t).position.km
                        vel2 = sat2.at(t).velocity.km_per_s

                        diff_pos = pos1 - pos2
                        dist = np.linalg.norm(diff_pos)

                        rel_vel = vel1 - vel2
                        if dist > 500.0:
                            radial_vel = np.dot(rel_vel, diff_pos) / dist
                            radial_vel = np.clip(radial_vel, -1.5, 1.5)
                        else:
                            radial_vel = 0.0

                        doppler = (radial_vel / 299792.458) * 30e9

                        all_times.append(t.utc_datetime())
                        all_distances.append(dist)
                        all_radial_vel.append(radial_vel)
                        all_doppler.append(doppler if dist < comm_threshold else np.nan)

                    progress = (current - start_date).total_seconds() / (end_date - start_date).total_seconds()
                    progress_bar.progress(min(int(progress*100), 100))
                    current = batch_end

                # 轉成陣列
                all_distances = np.array(all_distances)
                all_radial_vel = np.array(all_radial_vel)
                all_doppler = np.array(all_doppler)

                st.success("✅ 分析完成！")

                # 畫圖
                radial_vel_smooth = uniform_filter1d(all_radial_vel, size=smooth_window, mode='nearest')

                fig, axs = plt.subplots(3, 1, figsize=(16, 13), sharex=True)
                plt.subplots_adjust(hspace=0.4)

                axs[0].plot(all_times, all_distances, color='blue', linewidth=1.2, label='Distance')
                axs[0].axhline(y=comm_threshold, color='red', linestyle='--', linewidth=2, label=f'{comm_threshold} km Threshold')
                axs[0].set_ylabel('Distance (km)')
                axs[0].legend()
                axs[0].grid(True, alpha=0.3)

                axs[1].plot(all_times, radial_vel_smooth, color='orange', linewidth=1.3, label='Radial Relative Velocity (Smoothed)')
                axs[1].axhline(y=0, color='gray', linestyle='--')
                axs[1].set_ylabel('Radial Velocity (km/s)')
                axs[1].legend()
                axs[1].grid(True, alpha=0.3)

                dopp_plot = np.where(all_distances < comm_threshold, all_doppler/1000, np.nan)
                axs[2].plot(all_times, dopp_plot, color='green', linewidth=1.3, label='Doppler Shift (only in comm. range)')
                axs[2].axhline(y=0, color='gray', linestyle='--')
                axs[2].set_ylabel('Doppler Shift (kHz)')
                axs[2].set_xlabel('Time (UTC)')
                axs[2].legend()
                axs[2].grid(True, alpha=0.3)

                st.pyplot(fig)

                # 統計
                st.subheader("統計結果")
                mask = all_distances < comm_threshold
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("最小距離", f"{np.nanmin(all_distances):.2f} km")
                with col2:
                    st.metric("最大徑向速度", f"{np.nanmax(np.abs(all_radial_vel)):.4f} km/s")
                with col3:
                    st.metric("最大 Doppler", f"{np.nanmax(np.abs(all_doppler[mask]))/1000:.2f} kHz" if np.any(mask) else "N/A")

            except Exception as e:
                st.error(f"發生錯誤：{e}")

else:
    st.info("請上傳兩個 TLE 檔案後點擊開始分析")

st.caption("太空專題分析工具 | Streamlit 版本")