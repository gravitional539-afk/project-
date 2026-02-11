"""
수능 점수 예측 딥러닝 (신경망 시각화 + 입력)
실행: python3 main.py
"""
import warnings, os
import numpy as np
import matplotlib
matplotlib.set_loglevel("error")
warnings.filterwarnings("ignore", message=".*Glyph.*")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import fontManager
from matplotlib.widgets import TextBox, Button

for f in ["Apple SD Gothic Neo", "AppleGothic", "Malgun Gothic", "NanumGothic"]:
    if f in {x.name for x in fontManager.ttflist}:
        rcParams["font.family"] = f; break
rcParams["axes.unicode_minus"] = False
os.makedirs("output", exist_ok=True)

# ── 데이터 생성 & 학습 ──────────────────────────────────────
rng = np.random.RandomState(42)
N = 200
base = rng.uniform(40, 95, (N, 1))
X = np.clip(np.hstack([
    base + rng.normal(0, 8, (N,1)),
    base + rng.normal(0, 6, (N,1)),
    base + rng.normal(0, 4, (N,1)),
    base + rng.normal(0, 3, (N,1)),
]), 0, 100)
y = np.clip(0.1*X[:,[0]] + 0.2*X[:,[1]] + 0.3*X[:,[2]] + 0.4*X[:,[3]]
            + rng.normal(0, 2, (N,1)), 0, 100)

idx = rng.permutation(N); s = int(N*0.8)
X_tr, y_tr = X[idx[:s]], y[idx[:s]]
xmin, xmax = X_tr.min(0), X_tr.max(0)
ymin, ymax = y_tr.min(), y_tr.max()
Xn_tr = (X_tr-xmin)/(xmax-xmin+1e-8)
yn_tr = (y_tr-ymin)/(ymax-ymin+1e-8)
to_score = lambda yn: yn*(ymax-ymin)+ymin

relu = lambda z: np.maximum(0, z)
sizes = [4, 16, 8, 1]
W, B = [], []
for i in range(3):
    W.append(rng.randn(sizes[i], sizes[i+1]) * np.sqrt(2/sizes[i]))
    B.append(np.zeros((1, sizes[i+1])))

def forward(X):
    a = [X]
    for i in range(3):
        z = a[-1] @ W[i] + B[i]
        a.append(relu(z) if i < 2 else z)
    return a

print("학습 중...", end=" ")
for ep in range(300):
    a = forward(Xn_tr); m = len(yn_tr)
    da = 2*(a[3]-yn_tr)/m
    for i in [2,1,0]:
        dz = da * (a[i+1]>0).astype(float) if i < 2 else da
        W[i] -= 0.01 * (a[i].T @ dz)
        B[i] -= 0.01 * dz.sum(axis=0, keepdims=True)
        da = dz @ W[i].T
print("완료!")

# ── 신경망 구조도 + 입력 UI ──────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
fig.subplots_adjust(bottom=0.18)
ax.axis("off")
ax.set_xlim(-1, 5); ax.set_ylim(-1.5, 17)
ax.set_title("수능 예측 신경망  |  아래에 점수를 입력하세요",
             fontsize=15, fontweight="bold", pad=15)

layer_labels = ["입력층", "은닉층1\n(ReLU)", "은닉층2\n(ReLU)", "출력층"]
input_names = ["3월","6월","9월","11월"]
colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]
pos = {}

# 뉴런 그리기
circles = {}
for l, sz in enumerate(sizes):
    x = l * 1.4
    y_off = (max(sizes) - sz) / 2
    for n in range(sz):
        yy = y_off + n
        pos[(l,n)] = (x, yy)
        c = plt.Circle((x, yy), 0.3, color=colors[l], alpha=0.85,
                        ec="white", lw=2, zorder=5)
        ax.add_patch(c)
        circles[(l,n)] = c
        if l == 0:
            ax.text(x, yy, input_names[n], ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold", zorder=6)
        if l == 3:
            ax.text(x, yy, "수능", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold", zorder=6)
    ax.text(x, -1.2, f"{layer_labels[l]}\n({sz})", ha="center",
            fontsize=10, fontweight="bold")

# 연결선
lines = {}
for l in range(3):
    w = W[l]; wm = np.abs(w).max() + 1e-8
    for i in range(sizes[l]):
        for j in range(sizes[l+1]):
            x1,y1 = pos[(l,i)]; x2,y2 = pos[(l+1,j)]
            v = w[i,j]
            ln, = ax.plot([x1+0.3, x2-0.3], [y1, y2],
                          color="#1565C0" if v > 0 else "#C62828",
                          alpha=min(abs(v)/wm,1)*0.5+0.1,
                          lw=abs(v)/wm*2+0.3, zorder=1)
            lines[(l,i,j)] = ln

# 활성화 값 텍스트 (나중에 업데이트)
val_texts = {}
for l, sz in enumerate(sizes):
    for n in range(sz):
        x, yy = pos[(l,n)]
        t = ax.text(x, yy + 0.45, "", ha="center", va="bottom",
                    fontsize=6.5, color="#333", fontweight="bold", zorder=10)
        val_texts[(l,n)] = t

# 결과 텍스트
result_text = ax.text(4.3, max(sizes)/2, "", ha="left", va="center",
                      fontsize=18, fontweight="bold", color="#E91E63",
                      zorder=10)

ax.plot([],[], color="#1565C0", lw=2, label="양(+) 가중치")
ax.plot([],[], color="#C62828", lw=2, label="음(-) 가중치")
ax.legend(loc="upper right", fontsize=9)

# ── 입력 UI (하단) ───────────────────────────────────────────
month_labels = ["3월:", "6월:", "9월:", "11월:"]
defaults = ["70", "72", "75", "78"]
text_boxes = []

for i in range(4):
    bx = fig.add_axes([0.1 + i*0.17, 0.06, 0.1, 0.04])
    tb = TextBox(bx, month_labels[i], initial=defaults[i])
    text_boxes.append(tb)

def on_predict(_event=None):
    try:
        scores = [float(tb.text) for tb in text_boxes]
    except ValueError:
        result_text.set_text("숫자를 입력하세요")
        fig.canvas.draw_idle()
        return

    xn = (np.array([scores]) - xmin) / (xmax - xmin + 1e-8)
    a = forward(xn)
    pred = to_score(a[3]).flatten()[0]

    # 값 텍스트 업데이트
    for l in range(4):
        vals = a[l].flatten()
        for n in range(len(vals)):
            if l == 0:
                val_texts[(l,n)].set_text(f"{scores[n]:.0f}")
            elif l == 3:
                val_texts[(l,n)].set_text(f"{pred:.1f}")
            else:
                v = vals[n]
                val_texts[(l,n)].set_text(f"{v:.2f}" if v > 0 else "")

    # 활성화된 연결선 강조
    for l in range(3):
        a_in = a[l].flatten()
        a_out = a[l+1].flatten()
        for i in range(sizes[l]):
            for j in range(sizes[l+1]):
                active = a_in[i] > 0 and a_out[j] > 0
                ln = lines[(l,i,j)]
                ln.set_alpha(0.7 if active else 0.05)
                ln.set_linewidth(2.5 if active else 0.3)

    result_text.set_text(f"  {pred:.1f}점")
    fig.canvas.draw_idle()

btn_ax = fig.add_axes([0.8, 0.055, 0.1, 0.05])
btn = Button(btn_ax, "예측!", color="#E91E63", hovercolor="#C2185B")
btn.label.set_color("white")
btn.label.set_fontweight("bold")
btn.on_clicked(on_predict)

# 초기 예측 한번 실행
on_predict()
plt.show()
