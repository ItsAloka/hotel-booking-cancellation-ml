"""Draw the project workflow as one left-to-right picture.

Run from the project root:  .venv\\Scripts\\python scripts\\make_workflow_diagram.py
Writes reports/workflow_diagram.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / 'reports' / 'workflow_diagram.png'

# colours: one per kind of step
DATA, MODEL, TEST, CHECK, OUTPUT = '#DCE6F2', '#2B5C8F', '#E05A47', '#F2B84B', '#3C8D5A'

# (title, detail, fill colour, text colour) - main row, left to right
STEPS = [
    ('Raw data', '119,390 bookings\n32 columns\n2 Portuguese hotels', DATA, 'black'),
    ('Clean  (01)', 'remove 34,239\nduplicates, drop leak\ncolumns, fill gaps\n→ 84,969 bookings', DATA, 'black'),
    ('Split 80/20  (02)', 'stratified, seed 42\ntrain 67,975\ntest 16,994', DATA, 'black'),
    ('5 models  (03–07)', 'LogReg, Decision Tree,\nRandom Forest,\nXGBoost, Neural Net\ntuned with 5-fold CV', MODEL, 'white'),
    ('Pick best  (08)', 'rank on CV PR-AUC\n(training data only)\n→ XGBoost 0.7612', MODEL, 'white'),
    ('Test once  (08)', 'PR-AUC 0.7698\nrecall 0.769\naccuracy 0.819', TEST, 'white'),
    ('Checks  (09, 10)', 'weighting: no gain\nensemble: +0.0037,\nunder the noise\n→ keep XGBoost', CHECK, 'black'),
    ('Save model', 'models/xgboost.joblib\n(threshold 0.335)', OUTPUT, 'white'),
    ('Demo app', 'app/app.py\nscore one booking', OUTPUT, 'white'),
    ('Recommendation', 'flag risky bookings\nfor the revenue\nmanager', OUTPUT, 'white'),
]

BOX_W, BOX_H, STEP = 2.3, 2.1, 2.7
MAIN_Y = 3.4          # bottom edge of the main row
TEST_Y = 0.55         # bottom edge of the locked test-set box

fig, ax = plt.subplots(figsize=(26, 7.2))
ax.set_xlim(-0.4, STEP * len(STEPS) - 0.1)
ax.set_ylim(-0.2, 7.0)
ax.axis('off')


def box(x, y, w, h, fill, edge='#333', lw=1.5, style='round,pad=0.05,rounding_size=0.15', **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor=fill,
                                edgecolor=edge, linewidth=lw, **kw))


def arrow(start, end, color='#333', style='-|>', ls='-', lw=2.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=22, color=color,
                                 linewidth=lw, linestyle=ls, connectionstyle=f'arc3,rad={rad}'))


# lane backgrounds: the model steps that see training data only, and where the test set lives
box(3 * STEP - 0.2, MAIN_Y - 0.35, STEP + BOX_W + 0.4, BOX_H + 1.25, '#EEF3F9', edge='#9FB6D1', lw=1,
    style='round,pad=0,rounding_size=0.2', zorder=0)
ax.text(3 * STEP - 0.05, MAIN_Y + BOX_H + 0.62, 'TRAINING DATA ONLY  (67,975 bookings)',
        fontsize=13, fontweight='bold', color='#2B5C8F', va='center')

test_x0, test_x1 = 2 * STEP + 0.1, 5 * STEP + BOX_W
box(test_x0, TEST_Y, test_x1 - test_x0, 1.35, '#FBE3DF', edge=TEST, lw=2, zorder=1)
ax.text((test_x0 + test_x1) / 2, TEST_Y + 0.9, 'TEST SET  —  16,994 bookings, kept apart',
        ha='center', va='center', fontsize=14, fontweight='bold', color=TEST)
ax.text((test_x0 + test_x1) / 2, TEST_Y + 0.38,
        'not used for tuning, the threshold or picking the model — opened once, in 08',
        ha='center', va='center', fontsize=11, color='#7A2A20')

# main row
centres = []
for i, (title, detail, fill, colour) in enumerate(STEPS):
    x = i * STEP
    box(x, MAIN_Y, BOX_W, BOX_H, fill, zorder=2)
    ax.text(x + BOX_W / 2, MAIN_Y + BOX_H - 0.3, title, ha='center', va='center',
            fontsize=12, fontweight='bold', color=colour, zorder=3)
    ax.text(x + BOX_W / 2, MAIN_Y + BOX_H / 2 - 0.2, detail, ha='center', va='center',
            fontsize=9.5, color=colour, linespacing=1.35, zorder=3)
    ax.text(x + 0.12, MAIN_Y + 0.12, str(i + 1), fontsize=9, color=colour, alpha=0.8, zorder=3)
    centres.append(x + BOX_W / 2)
    if i:
        arrow((x - STEP + BOX_W + 0.03, MAIN_Y + BOX_H / 2), (x - 0.03, MAIN_Y + BOX_H / 2))

# the split sends 20% down into the locked box ...
arrow((centres[2], MAIN_Y - 0.03), (centres[2], TEST_Y + 1.38), color=TEST)
ax.text(centres[2] + 0.1, (MAIN_Y + TEST_Y + 1.35) / 2, '20% locked away',
        fontsize=10, color=TEST, va='center')
# ... and it only comes back up at the final test
arrow((centres[5], TEST_Y + 1.38), (centres[5], MAIN_Y - 0.03), color=TEST)
ax.text(centres[5] - 0.1, (MAIN_Y + TEST_Y + 1.35) / 2, 'opened once', fontsize=10,
        color=TEST, va='center', ha='right')
# notebook 10 reads it a second time, after XGBoost was already chosen
arrow((test_x1, TEST_Y + 0.7), (centres[6], MAIN_Y - 0.03), color=TEST, ls='--', lw=1.6, rad=0.25)
ax.text(centres[6] + 0.3, TEST_Y + 0.45, '10 re-reads it only\nto confirm, after\nthe choice was made',
        fontsize=9.5, color='#7A2A20', va='center')

ax.text(0, 6.75, 'Hotel booking cancellation — project workflow (notebooks 01–10)',
        fontsize=17, fontweight='bold', va='center')

fig.savefig(OUT, dpi=130, bbox_inches='tight', facecolor='white')
print('saved', OUT)
