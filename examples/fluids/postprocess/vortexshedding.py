import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.signal as sig


def coeff(force, rho=1, u=1, D=1, zspan=0.2):
    S = np.pi * D * zspan  # surface area
    return 2 * force / (rho * u**2 * S)


df = pandas.read_csv("force.csv")
df["Drag Coefficient"] = coeff(df["ForceX"])
df["Lift Coefficient"] = coeff(df["ForceY"])

sns.set_theme(style="ticks")
palette = sns.color_palette()
fig, ax_drag = plt.subplots()
ax_lift = ax_drag.twinx()

sns.lineplot(data=df, x="Time", y="Drag Coefficient",
             ax=ax_drag, color=palette[0])
sns.lineplot(data=df, x="Time", y="Lift Coefficient",
             ax=ax_lift, color=palette[1])
ax_drag.set_ylim(0.41, 0.48)
ax_drag.tick_params(axis="y", colors=palette[0])
ax_drag.yaxis.label.set_color(palette[0])
ax_lift.tick_params(axis="y", colors=palette[1])
ax_lift.yaxis.label.set_color(palette[1])

plt.show()


# compute and print shedding period
sample = df[df['Time'] > 70]    # once the initial transient has passed
peaks, _ = sig.find_peaks(sample['ForceY'])
period = np.diff(sample['Time'].iloc[peaks])
print(period)  # [5.55 5.6  5.55 5.55]
print(period.mean())  # 5.5625
