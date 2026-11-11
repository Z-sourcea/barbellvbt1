#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BarbellVBT 简洁增强版：训练视频分析 + 训练记录管理（实验原型）。"""

__version__ = "0.2.0"

import csv
import json
import math
import os
import threading
from datetime import datetime

import cv2
import numpy as np

from kivy.app import App
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from kivy.utils import platform as kivy_platform


COLOR_PRESETS = {
    "红/橙（推荐）": "red",
    "蓝色": "blue",
    "黄色": "yellow",
    "绿色": "green",
}

EXERCISES = ["深蹲", "前蹲", "硬拉", "高翻", "抓举", "卧推", "其他"]


def safe_float(value, default=None):
    try:
        number = float(str(value).strip())
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


class BarbellTracker:
    """按颜色标记追踪位置；返回 (x, y)，未识别时返回 None。"""

    def __init__(self, color="red"):
        self.color = color
        self.last_center = None

    def _mask(self, hsv):
        sat_low, val_low = 75, 55
        if self.color == "red":
            a = cv2.inRange(hsv, np.array([0, sat_low, val_low]), np.array([25, 255, 255]))
            b = cv2.inRange(hsv, np.array([170, sat_low, val_low]), np.array([180, 255, 255]))
            return cv2.bitwise_or(a, b)
        bounds = {
            "blue": ([90, sat_low, val_low], [135, 255, 255]),
            "yellow": ([20, sat_low, val_low], [38, 255, 255]),
            "green": ([38, sat_low, val_low], [88, 255, 255]),
        }
        low, high = bounds.get(self.color, bounds["blue"])
        return cv2.inRange(hsv, np.array(low), np.array(high))

    def detect(self, frame):
        if frame is None:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._mask(hsv)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 40:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            candidates.append((cx, cy, area))
        if not candidates:
            return None

        # 首帧选较大色块；之后优先选择附近的色块，降低跳到背景物体的机会。
        if self.last_center is None:
            chosen = max(candidates, key=lambda item: item[2])
        else:
            lx, ly = self.last_center
            chosen = max(
                candidates,
                key=lambda item: item[2] / (1.0 + math.hypot(item[0] - lx, item[1] - ly) / 120.0),
            )
        self.last_center = (chosen[0], chosen[1])
        return self.last_center


class VideoAnalyzer:
    """逐帧追踪标记，并估算连续运动片段的速度。"""

    def __init__(self, pixels_per_meter=1000.0, marker_color="red"):
        self.pixels_per_meter = float(pixels_per_meter)
        self.marker_color = marker_color
        self.results = []
        self.summary = {}

    @staticmethod
    def _smooth(values, window=3):
        if len(values) < 3:
            return values
        window = min(window, len(values) if len(values) % 2 else len(values) - 1)
        if window < 3:
            return values
        kernel = np.ones(window, dtype=float) / window
        padded = np.pad(values, (window // 2, window // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    def _segment_metrics(self, segment, number):
        times = np.asarray([p[0] for p in segment], dtype=float)
        ys = np.asarray([p[2] for p in segment], dtype=float)
        if len(times) < 4:
            return None
        elapsed = np.diff(times)
        if np.any(elapsed <= 0):
            return None
        y_m = ys / self.pixels_per_meter
        # np.gradient 使用实际时间戳，兼容漏检帧造成的不等间隔。
        signed_velocity = np.gradient(y_m, times)
        speeds = self._smooth(np.abs(signed_velocity), 3)
        if not np.all(np.isfinite(speeds)):
            return None
        duration = float(times[-1] - times[0])
        if duration <= 0:
            return None
        return {
            "rep": number,
            "peak_velocity": round(float(np.max(speeds)), 3),
            "mean_velocity": round(float(np.mean(speeds)), 3),
            "displacement": round(float((np.max(ys) - np.min(ys)) / self.pixels_per_meter), 3),
            "duration": round(duration, 2),
            "start_time": round(float(times[0]), 2),
            "end_time": round(float(times[-1]), 2),
            "tracked_frames": int(len(segment)),
        }

    def analyze(self, video_path, callback=None):
        if self.pixels_per_meter <= 0:
            return {"error": "标定值必须大于 0。"}
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "无法打开视频。请尝试 MP4 格式，并确认视频文件没有损坏。"}

        fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 30.0) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tracker = BarbellTracker(self.marker_color)
        points = []  # (time, x, y, frame_index)
        frame_count = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_count += 1
                center = tracker.detect(frame)
                if center is not None:
                    points.append((frame_count / fps, center[0], center[1], frame_count))
                if callback and frame_count % 30 == 0 and total_frames > 0:
                    callback({"progress": min(100.0, frame_count / total_frames * 100.0)})
        except Exception as exc:
            return {"error": f"读取视频时出错：{type(exc).__name__}: {exc}"}
        finally:
            cap.release()

        if len(points) < 6:
            return {
                "error": "有效标记点太少，无法计算速度。请检查标记颜色、光线、拍摄角度，或换一个更清晰的视频。"
            }

        # 先计算整段序列的速度，再按静止间隔分段。视频中若连续漏检超过约 0.35 秒，则切开，避免跨段计算。
        times = np.asarray([p[0] for p in points], dtype=float)
        ys = np.asarray([p[2] for p in points], dtype=float)
        y_m = ys / self.pixels_per_meter
        raw_speeds = np.abs(np.gradient(y_m, times))
        speeds = self._smooth(raw_speeds, 3)
        moving_threshold = 0.04  # m/s，实验性阈值
        stop_samples = max(6, int(round(fps * 0.25)))
        max_gap = 0.35
        segments = []
        active = []
        low_indices = []

        def close_segment():
            nonlocal active, low_indices
            # low_indices 单独缓存，没有追加到 active；只需丢弃缓存的静止尾帧。
            if len(active) >= 5:
                segments.append(active)
            active = []
            low_indices = []

        for i, point in enumerate(points):
            if i > 0 and times[i] - times[i - 1] > max_gap:
                close_segment()
            if speeds[i] >= moving_threshold:
                if low_indices and active:
                    # 短暂低速可能是动作转换/底部停顿，保留在动作片段中。
                    active.extend(low_indices)
                low_indices = []
                active.append(point)
            elif active:
                low_indices.append(point)
                if len(low_indices) >= stop_samples:
                    close_segment()
        close_segment()

        results = []
        for segment in segments:
            item = self._segment_metrics(segment, len(results) + 1)
            if item:
                results.append(item)
        if not results:
            return {
                "error": "没有识别到足够长的运动片段。可以尝试降低遮挡、固定手机，并让标记在画面中更明显。"
            }

        # 每个动作段与本视频中最快动作段作比较，显示速度下降比例；这不是疲劳诊断。
        best_mean = max(item["mean_velocity"] for item in results)
        for item in results:
            item["velocity_loss_pct"] = round(
                max(0.0, (best_mean - item["mean_velocity"]) / best_mean * 100.0), 1
            ) if best_mean > 0 else 0.0

        coverage = min(100.0, len(points) / max(frame_count, 1) * 100.0)
        self.results = results
        self.summary = {
            "fps": round(float(fps), 2),
            "total_frames": int(frame_count),
            "tracked_frames": int(len(points)),
            "tracking_coverage_pct": round(float(coverage), 1),
            "mean_velocity": round(float(np.mean([r["mean_velocity"] for r in results])), 3),
            "peak_velocity": round(float(max(r["peak_velocity"] for r in results)), 3),
            "best_mean_velocity": round(float(best_mean), 3),
            "last_velocity_loss_pct": results[-1]["velocity_loss_pct"] if len(results) >= 2 else None,
            "segment_count": len(results),
        }
        if callback:
            callback({"progress": 100.0})
        return {"results": results, "summary": self.summary}

    @staticmethod
    def export_csv(filepath, session, results, summary):
        with open(filepath, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["BarbellVBT 训练视频分析记录"])
            writer.writerow(["日期", session.get("date", "")])
            writer.writerow(["训练者", session.get("athlete", "")])
            writer.writerow(["动作", session.get("exercise", "")])
            writer.writerow(["负荷(kg)", session.get("load_kg", "")])
            writer.writerow(["标记颜色", session.get("marker_color", "")])
            writer.writerow(["标定值(px/m)", session.get("pixels_per_meter", "")])
            writer.writerow(["视频", session.get("video_name", "")])
            writer.writerow(["识别覆盖率(%)", summary.get("tracking_coverage_pct", "")])
            writer.writerow(["平均速度(m/s)", summary.get("mean_velocity", "")])
            writer.writerow(["峰值速度(m/s)", summary.get("peak_velocity", "")])
            writer.writerow([])
            writer.writerow([
                "动作段", "峰值速度(m/s)", "平均速度(m/s)", "位移(m)",
                "持续时间(s)", "开始时间(s)", "结束时间(s)", "相对最快段速度下降(%)",
            ])
            for item in results:
                writer.writerow([
                    item["rep"], item["peak_velocity"], item["mean_velocity"],
                    item["displacement"], item["duration"], item["start_time"],
                    item["end_time"], item["velocity_loss_pct"],
                ])
        return filepath


class BarbellVBTApp(App):
    def build(self):
        self.title = "BarbellVBT 杠铃速度分析"
        self.pixels_per_meter = 1000.0
        self.marker_label = "红/橙（推荐）"
        self.analyzer = VideoAnalyzer(self.pixels_per_meter, "red")
        self.results = []
        self.summary = {}
        self.selected_video = None
        self.session = {}

        root = BoxLayout(orientation="vertical", padding=dp(10), spacing=dp(7))
        self.scroll = ScrollView()
        content = BoxLayout(orientation="vertical", spacing=dp(8), size_hint_y=None, padding=(0, 0, 0, dp(8)))
        content.bind(minimum_height=content.setter("height"))

        title = Label(
            text="BARBELL VBT\n杠铃速度分析 · 增强简洁版",
            font_size="21sp", bold=True, size_hint_y=None, height=dp(70),
        )
        content.add_widget(title)
        subtitle = Label(
            text="上传训练视频 · 分段估算速度 · 记录训练数据",
            font_size="13sp", size_hint_y=None, height=dp(28),
        )
        content.add_widget(subtitle)

        content.add_widget(self._section_label("01 训练信息"))
        self.athlete_input = self._input("训练者（可不填）", height=42)
        content.add_widget(self.athlete_input)

        row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        self.exercise_spinner = Spinner(text="深蹲", values=EXERCISES, size_hint_x=0.58)
        self.load_input = self._input("负荷 kg（可不填）", height=44, input_filter="float")
        row.add_widget(self.exercise_spinner)
        row.add_widget(self.load_input)
        content.add_widget(row)
        self.notes_input = TextInput(
            hint_text="训练备注（可不填，例如 RPE、训练计划）",
            multiline=True, size_hint_y=None, height=dp(58),
        )
        content.add_widget(self.notes_input)

        content.add_widget(self._section_label("02 视频与识别设置"))
        self.marker_spinner = Spinner(text=self.marker_label, values=list(COLOR_PRESETS.keys()), size_hint_y=None, height=dp(44))
        content.add_widget(self.marker_spinner)
        self.select_btn = Button(text="① 选择训练视频", size_hint_y=None, height=dp(48), font_size="16sp")
        self.select_btn.bind(on_press=self.open_filechooser)
        content.add_widget(self.select_btn)
        self.video_label = Label(text="还没有选择视频", size_hint_y=None, height=dp(30), font_size="12sp")
        content.add_widget(self.video_label)

        row2 = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(6))
        self.analyze_btn = Button(text="② 开始分析", disabled=True, font_size="16sp")
        self.analyze_btn.bind(on_press=self.start_analysis)
        row2.add_widget(self.analyze_btn)
        calibrate_btn = Button(text="标定设置", font_size="14sp")
        calibrate_btn.bind(on_press=self.open_calibration)
        row2.add_widget(calibrate_btn)
        content.add_widget(row2)

        self.status_label = Label(
            text="提示：把鲜明颜色标记贴在杠铃片上，手机固定在侧面拍摄。",
            size_hint_y=None, height=dp(56), font_size="12sp", halign="left", valign="middle",
        )
        self._wrap_label(self.status_label)
        content.add_widget(self.status_label)

        content.add_widget(self._section_label("03 本次分析摘要"))
        self.metrics_grid = GridLayout(cols=2, spacing=dp(6), size_hint_y=None, height=dp(112))
        self.metric_labels = {}
        for key, caption in [
            ("avg", "片段平均速度"), ("peak", "最高峰值速度"),
            ("count", "识别运动片段"), ("coverage", "标记识别覆盖率"),
        ]:
            box = BoxLayout(orientation="vertical", padding=dp(6), spacing=dp(2))
            box.add_widget(Label(text=caption, font_size="11sp"))
            value = Label(text="—", font_size="18sp", bold=True)
            box.add_widget(value)
            self.metric_labels[key] = value
            self.metrics_grid.add_widget(box)
        content.add_widget(self.metrics_grid)
        self.loss_label = Label(text="速度变化：等待分析", size_hint_y=None, height=dp(42), font_size="13sp", halign="left")
        self._wrap_label(self.loss_label)
        content.add_widget(self.loss_label)
        self.advice_label = Label(
            text="分析提示会在完成后显示。请记住：这是视频估算，不是专业测力设备。",
            size_hint_y=None, height=dp(68), font_size="12sp", halign="left", valign="middle",
        )
        self._wrap_label(self.advice_label)
        content.add_widget(self.advice_label)

        content.add_widget(self._section_label("04 每个动作段的结果"))
        self.result_layout = BoxLayout(orientation="vertical", spacing=dp(4), size_hint_y=None)
        self.result_layout.bind(minimum_height=self.result_layout.setter("height"))
        content.add_widget(self.result_layout)

        content.add_widget(self._section_label("05 保存与导出"))
        row3 = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(6))
        self.save_btn = Button(text="保存训练记录", disabled=True)
        self.save_btn.bind(on_press=self.save_session)
        row3.add_widget(self.save_btn)
        self.export_btn = Button(text="导出 CSV", disabled=True)
        self.export_btn.bind(on_press=self.export_results)
        row3.add_widget(self.export_btn)
        content.add_widget(row3)
        history_btn = Button(text="查看本机历史记录", size_hint_y=None, height=dp(44))
        history_btn.bind(on_press=self.show_history)
        content.add_widget(history_btn)
        clear_btn = Button(text="清空当前结果", size_hint_y=None, height=dp(40))
        clear_btn.bind(on_press=self.clear_results)
        content.add_widget(clear_btn)

        warning = Label(
            text="重要：结果受拍摄角度、标记识别、透视和光线影响；请人工核对视频。暂不用于专业负荷决策。",
            font_size="11sp", size_hint_y=None, height=dp(54),
        )
        content.add_widget(warning)

        self.scroll.add_widget(content)
        root.add_widget(self.scroll)
        return root

    @staticmethod
    def _wrap_label(label):
        label.bind(size=lambda obj, size: setattr(obj, "text_size", (max(size[0] - dp(8), 0), None)))
        return label

    @staticmethod
    def _section_label(text):
        return Label(text=text, bold=True, halign="left", size_hint_y=None, height=dp(30), font_size="15sp")

    @staticmethod
    def _input(hint, height=42, input_filter=None):
        field = TextInput(hint_text=hint, multiline=False, size_hint_y=None, height=dp(height))
        if input_filter:
            field.input_filter = input_filter
        return field

    def on_start(self):
        if kivy_platform == "android":
            try:
                from android.permissions import request_permissions
                from jnius import autoclass
                sdk = autoclass("android.os.Build$VERSION").SDK_INT
                permission = (
                    "android.permission.READ_MEDIA_VIDEO" if sdk >= 33
                    else "android.permission.READ_EXTERNAL_STORAGE"
                )
                request_permissions([permission])
            except Exception as exc:
                self.status_label.text = f"视频权限申请提示：{exc}"

    def open_filechooser(self, _instance):
        start_path = "/storage/emulated/0/" if kivy_platform == "android" else os.path.expanduser("~")
        chooser = FileChooserListView(
            path=start_path,
            filters=["*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mov", "*.MOV", "*.mkv", "*.MKV", "*.3gp", "*.3GP"],
        )
        buttons = BoxLayout(size_hint_y=0.13, spacing=dp(6))
        ok_btn = Button(text="选中这个视频")
        cancel_btn = Button(text="取消")
        buttons.add_widget(ok_btn)
        buttons.add_widget(cancel_btn)
        box = BoxLayout(orientation="vertical", spacing=dp(6))
        box.add_widget(chooser)
        box.add_widget(buttons)
        popup = Popup(title="选择训练视频", content=box, size_hint=(0.95, 0.92))

        def confirm(_btn):
            if chooser.selection:
                self.selected_video = chooser.selection[0]
                self.video_label.text = f"已选：{os.path.basename(self.selected_video)}"
                self.analyze_btn.disabled = False
            popup.dismiss()

        ok_btn.bind(on_press=confirm)
        cancel_btn.bind(on_press=popup.dismiss)
        popup.open()

    def open_calibration(self, _instance):
        box = BoxLayout(orientation="vertical", padding=dp(10), spacing=dp(8))
        box.add_widget(Label(
            text="输入像素/米（px/m）。例如：45 cm 杠铃片在画面中宽 90 px，则 90 ÷ 0.45 = 200 px/m。\n\n建议使用与杠铃运动处于同一平面的已知长度进行标定。",
            font_size="13sp",
        ))
        field = TextInput(text=str(self.pixels_per_meter), multiline=False, input_filter="float", size_hint_y=None, height=dp(44))
        box.add_widget(field)
        save = Button(text="保存", size_hint_y=None, height=dp(42))
        box.add_widget(save)
        popup = Popup(title="标定设置", content=box, size_hint=(0.9, 0.55))

        def save_value(_btn):
            value = safe_float(field.text)
            if value is None or value <= 0:
                self.status_label.text = "标定值必须是大于 0 的数字。"
                return
            self.pixels_per_meter = value
            self.status_label.text = f"标定值已保存：{value:g} px/m"
            popup.dismiss()

        save.bind(on_press=save_value)
        popup.open()

    def start_analysis(self, _instance):
        if not self.selected_video:
            self.status_label.text = "请先选择训练视频。"
            return
        load = safe_float(self.load_input.text, None)
        if self.load_input.text.strip() and (load is None or load < 0):
            self.status_label.text = "负荷请输入大于或等于 0 的数字，或留空。"
            return

        marker_label = self.marker_spinner.text
        marker_color = COLOR_PRESETS.get(marker_label, "red")
        self.analyzer = VideoAnalyzer(self.pixels_per_meter, marker_color)
        self.marker_label = marker_label
        self.results = []
        self.summary = {}
        self.result_layout.clear_widgets()
        for value in self.metric_labels.values():
            value.text = "—"
        self.loss_label.text = "速度变化：分析中……"
        self.advice_label.text = "正在分析视频，请稍等。"
        self.save_btn.disabled = True
        self.export_btn.disabled = True
        self.analyze_btn.disabled = True
        self.select_btn.disabled = True
        self.status_label.text = "正在逐帧分析视频……"
        self.session = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "athlete": self.athlete_input.text.strip(),
            "exercise": self.exercise_spinner.text,
            "load_kg": load,
            "marker_color": marker_label,
            "pixels_per_meter": self.pixels_per_meter,
            "video_name": os.path.basename(self.selected_video),
            "notes": self.notes_input.text.strip(),
        }
        threading.Thread(target=self._worker, args=(self.selected_video,), daemon=True).start()

    def _worker(self, path):
        def callback(data):
            if "progress" in data:
                p = data["progress"]
                Clock.schedule_once(lambda _dt, value=p: self._set_progress(value))
        try:
            result = self.analyzer.analyze(path, callback)
        except Exception as exc:
            result = {"error": f"分析异常：{type(exc).__name__}: {exc}"}
        Clock.schedule_once(lambda _dt, value=result: self._finish_analysis(value))

    def _set_progress(self, progress):
        self.status_label.text = f"正在分析视频……{progress:.0f}%"

    def _finish_analysis(self, result):
        self.analyze_btn.disabled = False
        self.select_btn.disabled = False
        if isinstance(result, dict) and "error" in result:
            self.status_label.text = result["error"]
            self.loss_label.text = "速度变化：没有可用结果"
            self.advice_label.text = "请检查视频、颜色标记、光线和标定值后重试。"
            return
        self.results = result["results"]
        self.summary = result["summary"]
        self.metric_labels["avg"].text = f"{self.summary['mean_velocity']:.3f} m/s"
        self.metric_labels["peak"].text = f"{self.summary['peak_velocity']:.3f} m/s"
        self.metric_labels["count"].text = str(self.summary["segment_count"])
        coverage = self.summary["tracking_coverage_pct"]
        self.metric_labels["coverage"].text = f"{coverage:.1f}%"

        if self.summary["last_velocity_loss_pct"] is None:
            self.loss_label.text = "速度变化：仅识别到一个动作段，暂不能比较"
        else:
            self.loss_label.text = f"最后动作段较最快动作段下降：{self.summary['last_velocity_loss_pct']:.1f}%"

        self.advice_label.text = self._make_advice(coverage, self.summary["last_velocity_loss_pct"])
        self.result_layout.clear_widgets()
        for item in self.results:
            row = BoxLayout(size_hint_y=None, height=dp(64), spacing=dp(4))
            text = (
                f"动作段 {item['rep']}  |  平均 {item['mean_velocity']:.3f} m/s  |  峰值 {item['peak_velocity']:.3f} m/s\n"
                f"位移 {item['displacement']:.3f} m  |  时长 {item['duration']:.2f} s  |  相对下降 {item['velocity_loss_pct']:.1f}%"
            )
            result_label = Label(text=text, font_size="12sp", halign="left", valign="middle")
            self._wrap_label(result_label)
            row.add_widget(result_label)
            self.result_layout.add_widget(row)
        self.status_label.text = f"分析完成：得到 {len(self.results)} 个连续运动片段。请人工回看核对。"
        self.save_btn.disabled = False
        self.export_btn.disabled = False

    @staticmethod
    def _make_advice(coverage, loss):
        if coverage < 45:
            return "识别覆盖率偏低：可能有遮挡、标记颜色不明显或光线不足。本次速度结果可信度较低，请重新拍摄。"
        if coverage < 75:
            base = "识别覆盖率一般，请检查是否有漏检或误识别。"
        else:
            base = "标记识别覆盖率较高，但仍需人工确认追踪点是否始终跟随杠铃。"
        if loss is None:
            return base + " 只有一个动作段，暂时无法比较速度变化。"
        if loss < 10:
            return base + " 最后动作段相对最快段的速度下降较小。"
        if loss < 20:
            return base + " 片段间速度有一定下降，建议回看视频确认动作质量。"
        return base + " 片段间速度下降较明显，建议回看视频确认分段和标记识别，不要仅凭这个数值决定训练负荷。"

    def _history_path(self):
        return os.path.join(self.user_data_dir, "training_history.json")

    def save_session(self, _instance):
        if not self.results:
            self.status_label.text = "目前没有可保存的分析结果。"
            return
        record = {
            **self.session,
            "summary": self.summary,
            "results": self.results,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        path = self._history_path()
        try:
            history = []
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as file:
                    loaded = json.load(file)
                    if isinstance(loaded, list):
                        history = loaded
            history.append(record)
            history = history[-100:]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            temp_path = path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as file:
                json.dump(history, file, ensure_ascii=False, indent=2)
            os.replace(temp_path, path)
            self.status_label.text = f"已保存训练记录。本机共保存 {len(history)} 次。"
        except Exception as exc:
            self.status_label.text = f"保存失败：{type(exc).__name__}: {exc}"

    def show_history(self, _instance):
        path = self._history_path()
        try:
            if not os.path.exists(path):
                content = "还没有保存过训练记录。分析完成后点击‘保存训练记录’即可。"
            else:
                with open(path, "r", encoding="utf-8") as file:
                    history = json.load(file)
                lines = []
                for item in reversed(history[-20:]):
                    summary = item.get("summary", {})
                    lines.append(
                        f"{item.get('saved_at', item.get('date', ''))} | {item.get('exercise', '')} | "
                        f"负荷 {item.get('load_kg') or '未填'} kg | "
                        f"平均 {summary.get('mean_velocity', '—')} m/s | "
                        f"{summary.get('segment_count', 0)} 段"
                    )
                content = "\n\n".join(lines) if lines else "历史记录为空。"
        except Exception as exc:
            content = f"读取历史记录失败：{type(exc).__name__}: {exc}"
        scroll = ScrollView()
        label = Label(text=content, font_size="12sp", halign="left", valign="top", size_hint_y=None)
        label.bind(size=lambda obj, size: setattr(obj, "text_size", (max(size[0] - dp(10), 0), None)))
        label.bind(texture_size=lambda obj, size: setattr(obj, "height", size[1] + dp(20)))
        scroll.add_widget(label)
        popup = Popup(title="最近 20 次训练记录", content=scroll, size_hint=(0.92, 0.78))
        popup.open()

    def export_results(self, _instance):
        if not self.results:
            self.status_label.text = "没有可导出的结果。"
            return
        filename = f"BarbellVBT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            if kivy_platform == "android":
                out_dir = os.path.join(self.user_data_dir, "exports")
            else:
                out_dir = os.path.join(os.path.expanduser("~"), "Downloads")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, filename)
            VideoAnalyzer.export_csv(path, self.session, self.results, self.summary)
            message = f"CSV 已导出：\n{path}\n\nAndroid 版本暂时保存在 App 私有文件夹内。"
            self.status_label.text = f"CSV 已导出：{filename}"
        except Exception as exc:
            message = f"导出失败：{type(exc).__name__}: {exc}"
            self.status_label.text = "CSV 导出失败。"
        Popup(title="导出结果", content=Label(text=message, font_size="12sp"), size_hint=(0.9, 0.38)).open()

    def clear_results(self, _instance):
        self.results = []
        self.summary = {}
        self.result_layout.clear_widgets()
        for label in self.metric_labels.values():
            label.text = "—"
        self.loss_label.text = "速度变化：等待分析"
        self.advice_label.text = "分析提示会在完成后显示。"
        self.save_btn.disabled = True
        self.export_btn.disabled = True
        self.status_label.text = "当前结果已清空；已保存的历史记录不会删除。"


def run_desktop_mode():
    print("BarbellVBT 增强简洁版：桌面简易测试模式")
    video = input("把训练视频拖到这里并按回车：").strip().strip('"')
    if not os.path.isfile(video):
        print("找不到视频文件。")
        return
    raw_ppm = input("标定值 px/m（默认 1000）：").strip()
    ppm = safe_float(raw_ppm, 1000.0)
    if ppm is None or ppm <= 0:
        print("标定值无效。")
        return
    analyzer = VideoAnalyzer(ppm, "red")
    result = analyzer.analyze(video, lambda d: print(f"分析进度：{d['progress']:.0f}%") if "progress" in d else None)
    if "error" in result:
        print(result["error"])
        return
    print("摘要：", result["summary"])
    for row in result["results"]:
        print(row)
    session = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "athlete": "", "exercise": "未指定", "load_kg": None,
        "marker_color": "红/橙（推荐）", "pixels_per_meter": ppm,
        "video_name": os.path.basename(video),
    }
    output = f"BarbellVBT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    VideoAnalyzer.export_csv(output, session, result["results"], result["summary"])
    print(f"CSV 已保存：{os.path.abspath(output)}")


if __name__ == "__main__":
    if kivy_platform == "android":
        BarbellVBTApp().run()
    else:
        run_desktop_mode()
