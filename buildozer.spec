[app]

# 手机上显示的名称
title = BarbellVBT
package.name = barbellvbt
package.domain = com.barbellvbt

source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,atlas,txt,md
source.exclude_exts = spec
source.exclude_dirs = .git,.github,.buildozer,bin,venv,__pycache__

version = 0.2.0

# OpenCV 必须使用 python-for-android 的 opencv recipe，不能写 opencv-python。
requirements = python3==3.11.9,kivy==2.3.0,numpy,opencv

# Android 视频读取权限：Android 13+ 使用 READ_MEDIA_VIDEO，旧系统使用 READ_EXTERNAL_STORAGE。
android.permissions = READ_MEDIA_VIDEO,(name=android.permission.READ_EXTERNAL_STORAGE;maxSdkVersion=32)

android.api = 35
# 当前 python-for-android NumPy recipe 最低要求 API 24
android.minapi = 24
android.ndk = 28c
android.accept_sdk_license = True
android.archs = arm64-v8a

icon.filename = assets/icon.png
presplash.filename = assets/splash.png
presplash_color = #101820

orientation = portrait
fullscreen = 0
android.logcat_filters = *:S python:D

# python-for-android develop 分支包含当前 OpenCV recipe。
p4a.branch = develop

[buildozer]
log_level = 2
warn_on_root = 0
