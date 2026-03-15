import os
import zipfile
import tempfile
import numpy as np
import pandas as pd
import xarray as xr

def build_sample_numpy(zip_path, ibtracs_path, target_time, radius_deg=3.0):
    """
    生成单时间步的模型输入矩阵 (X) 和 标签矩阵 (Y)，纯 NumPy 实现
    """
    print(f"\n🚀 开始构建时间步 [{target_time}] 的 NumPy 矩阵...")
    
    # 设定目标经纬度范围 (0-60°N, 100-180°E)
    target_lats = slice(60.0, 0.0) # ECMWF 纬度通常降序
    target_lons = slice(100.0, 180.0)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. 解压所需文件
        print("   -> 正在从 zip 中提取 .nc 文件...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            nc_files = [f for f in z.namelist() if f.endswith('.nc')]
            oper_file = next(f for f in nc_files if 'oper' in f) # 气象/风场
            wave_file = next(f for f in nc_files if 'wave' in f) # 海浪场
            
            z.extract(oper_file, path=temp_dir)
            z.extract(wave_file, path=temp_dir)
            
            oper_path = os.path.join(temp_dir, oper_file)
            wave_path = os.path.join(temp_dir, wave_file)

        # 2. 读取并截取空间与时间 (构建 X)
        print("   -> 正在截取西太平洋海区 (0-60°N, 100-180°E)...")
        ds_wind = xr.open_dataset(oper_path).sel(
            latitude=target_lats, longitude=target_lons, valid_time=target_time
        )
        ds_wave = xr.open_dataset(wave_path).sel(
            latitude=target_lats, longitude=target_lons, valid_time=target_time
        )
        
        # 3. 核心：空间对齐 (将 0.5° 的浪场插值到 0.25° 的风场网格上)
        print("   -> 正在执行双线性插值 (0.5° -> 0.25°)...")
        ds_wave_interp = ds_wave.interp(
            latitude=ds_wind.latitude, 
            longitude=ds_wind.longitude, 
            method='linear'
        )
        
        # 4. 提取 5 个物理量数组 (并将陆地上的海浪 NaN 值填充为 0)
        u10 = ds_wind['u10'].values
        v10 = ds_wind['v10'].values
        mwd = np.nan_to_num(ds_wave_interp['mwd'].values, nan=0.0)
        mwp = np.nan_to_num(ds_wave_interp['mwp'].values, nan=0.0)
        swh = np.nan_to_num(ds_wave_interp['swh'].values, nan=0.0)
        
        # 拼接成 (Channels, Height, Width) 的 NumPy 数组
        X_array = np.stack([u10, v10, mwd, mwp, swh], axis=0).astype(np.float32)
        
        # 记录网格坐标供打标签用
        lat_grid = ds_wind.latitude.values
        lon_grid = ds_wind.longitude.values
        
        ds_wind.close()
        ds_wave.close()
        
    # 5. 生成真实标签 (构建 Y)
    print("   -> 正在查询 IBTrACS 台风轨迹打标签...")
    Y_array = np.zeros((1, len(lat_grid), len(lon_grid)), dtype=np.float32)
    
    ds_ibtracs = xr.open_dataset(ibtracs_path)
    target_dt = np.datetime64(target_time)
    
    times = ds_ibtracs['time'].values
    match_indices = np.where(times == target_dt)
    
    if len(match_indices[0]) > 0:
        lats_ib = ds_ibtracs['lat'].values
        lons_ib = ds_ibtracs['lon'].values
        Lon_mesh, Lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        typhoon_found = False
        for s_id, t_id in zip(match_indices[0], match_indices[1]):
            center_lat = lats_ib[s_id, t_id]
            center_lon = lons_ib[s_id, t_id]
            
            if not np.isnan(center_lat) and not np.isnan(center_lon):
                # 检查台风是否在我们的西太平洋网格内
                if (0 <= center_lat <= 60) and (100 <= center_lon <= 180):
                    dist = np.sqrt((Lat_mesh - center_lat)**2 + (Lon_mesh - center_lon)**2)
                    Y_array[0, dist <= radius_deg] = 1.0
                    typhoon_found = True
                    print(f"      🎯 [发现台风] 中心坐标: ({center_lat:.1f}°N, {center_lon:.1f}°E)")
                    
        if not typhoon_found:
             print("      (台风存在，但中心不在西太平洋目标海区内)")
    else:
        print("      (此时刻整个太平洋无记录台风，此样本为纯负样本)")
        
    ds_ibtracs.close()
    
    return X_array, Y_array

# ================= 测试运行 =================
if __name__ == "__main__":
    # 请根据你的实际路径调整
    test_zip = r"D:\\oceanSys\\wavedata\\2016\\201602.zip"
    ibtracs_file = r"D:\\oceanSys\\IBTrACS.WP.v04r01.nc"
    
    test_time = '2016-02-01T00:00:00' 
    
    try:
        X_mat, Y_mat = build_sample_numpy(test_zip, ibtracs_file, test_time)
        
        print("\n" + "="*50)
        print("✅ 数据矩阵构建成功！(纯 NumPy)")
        print("="*50)
        
        print("【输入特征矩阵 X (风浪 5 变量)】")
        print(f"  - 形状 (Shape): {X_mat.shape}  -> (通道数, 纬度 H, 经度 W)")
        print(f"  - 数据类型: {X_mat.dtype}")
        print(f"  - 通道顺序: [U10, V10, MWD, MWP, SWH]")
        print(f"  - SWH (有效波高) 最大值: {np.max(X_mat[4]):.2f} 米")
        print(f"  - SWH (有效波高) 最小值: {np.min(X_mat[4]):.2f} 米 (0代表陆地)")
        
        print("\n【目标标签矩阵 Y (台风 Mask)】")
        print(f"  - 形状 (Shape): {Y_mat.shape}  -> (1, 纬度 H, 经度 W)")
        print(f"  - 异常网格点数量: {int(np.sum(Y_mat))} / {Y_mat.shape[1] * Y_mat.shape[2]}")
        print("="*50)
        
    except Exception as e:
        print(f"运行失败: {e}")