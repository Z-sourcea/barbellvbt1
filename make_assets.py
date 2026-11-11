"""重新生成 App 图标和启动画面；正常打包时无需运行。"""
import os
from PIL import Image, ImageDraw

os.makedirs("assets", exist_ok=True)

# App 图标
img = Image.new("RGBA", (512, 512), (16, 24, 32, 255))
d = ImageDraw.Draw(img)
d.rounded_rectangle((42, 42, 470, 470), radius=95, fill=(26, 108, 88, 255))
y = 256
d.rounded_rectangle((94, y - 12, 418, y + 12), radius=12, fill=(235, 240, 244, 255))
d.rounded_rectangle((110, y - 86, 154, y + 86), radius=12, fill=(235, 240, 244, 255))
d.rounded_rectangle((164, y - 62, 191, y + 62), radius=8, fill=(245, 145, 60, 255))
d.rounded_rectangle((358, y - 62, 385, y + 62), radius=8, fill=(245, 145, 60, 255))
d.rounded_rectangle((402, y - 86, 446, y + 86), radius=12, fill=(235, 240, 244, 255))
img.save("assets/icon.png")

# 启动画面
splash = Image.new("RGBA", (1080, 1920), (16, 24, 32, 255))
d = ImageDraw.Draw(splash)
d.rounded_rectangle((150, 690, 930, 1230), radius=90, fill=(26, 108, 88, 255))
y = 960
d.rounded_rectangle((280, y - 18, 800, y + 18), radius=18, fill=(235, 240, 244, 255))
for x, w, h in [(315, 56, 210), (385, 42, 160), (695, 42, 160), (765, 56, 210)]:
    d.rounded_rectangle((x - w // 2, y - h // 2, x + w // 2, y + h // 2), radius=12, fill=(245, 145, 60, 255))
splash.save("assets/splash.png")
print("图标和启动画面已生成。")
