"""Region coordinates per film, in pixels of the saved source file.

Each entry is source filename -> (number within its regime, regions). Coordinates for the five
films below are the ones recorded in the medical-examples card; the rest were read from the saved
file in this pass.
"""

SPECS = {
    "silicotuberculosis-cavity-and-large-opacity-pyana-kitenge-2025-fig1-case2.png": (1, [
        dict(shape="circle", cx=135, cy=133, r=22, label="TB cavity",
             label_at=(135, 100), label_anchor="ms"),
        dict(shape="circle", cx=228, cy=108, r=28, label="silicosis large opacity",
             label_at=(228, 66), label_anchor="ms"),
    ]),
    "comorbidity-pneumothorax-and-nodule-nih-00017243-010.png": (4, [
        dict(shape="box", x=21.7, y=291.5, w=124.6, h=564.6, label="pneumothorax",
             label_at=(158, 300), label_anchor="ls"),
        dict(shape="box", x=76.9, y=602.5, w=88.9, h=83.4, label="nodule",
             label_at=(180, 660), label_anchor="ls"),
    ]),
    "superimposed-aspergilloma-in-tb-cavity-bongomin-2020-fig3a.jpg": (1, [
        dict(shape="ellipse", bbox=[544, 226, 627, 314], label="old TB cavity wall",
             label_at=(586, 196), label_anchor="ms"),
        dict(shape="circle", cx=594, cy=262, r=30, label="aspergilloma inside it",
             label_at=(700, 360), label_anchor="ms"),
    ]),
    "superimposed-pneumonia-in-emphysema-pata-2021-fig2.jpg": (2, [
        dict(shape="polygon", points=[(447, 345), (470, 335), (510, 340), (530, 370), (545, 410),
                                      (540, 440), (510, 455), (470, 450), (450, 420), (445, 380)],
             label="pneumonia, broken up by the holes", label_at=(495, 500), label_anchor="ms"),
        dict(shape="circle", cx=548, cy=307, r=38, label="emphysematous space",
             label_at=(548, 255), label_anchor="ms"),
    ]),
    "superimposed-cancer-in-fibrosis-kim-2026-fig2a.png": (1, [
        dict(shape="polygon", points=[(530, 800), (600, 650), (700, 620), (830, 620), (920, 650),
                                      (935, 780), (900, 900), (860, 960), (700, 975), (560, 950),
                                      (525, 880)],
             label="carcinoma, read as pneumonia", label_at=(730, 596), label_anchor="ms"),
    ]),
}

SPECS.update({
    "comorbidity-copd-and-calcified-granuloma-ibraheem-2024-fig1.jpg": (2, [
        dict(shape="circle", cx=572, cy=180, r=16, label="calcified granuloma",
             label_at=(572, 132), label_anchor="ms"),
        dict(shape="polygon", points=[(80, 650), (150, 630), (230, 616), (300, 608),
                                      (300, 585), (230, 593), (150, 607), (80, 627)],
             label="flattened diaphragm (COPD)", label_at=(210, 706), label_anchor="ms"),
    ]),
    "cardiomegaly-and-effusion-heart-failure-palmeira-2016-fig3.jpg": (3, [
        dict(shape="polygon", points=[(168, 148), (205, 154), (236, 184), (254, 220), (249, 248),
                                      (205, 262), (160, 266), (126, 248), (110, 216), (122, 178),
                                      (142, 156)],
             label="enlarged heart", label_at=(183, 136), label_anchor="ms"),
        dict(shape="polygon", points=[(40, 242), (108, 246), (112, 292), (44, 296)],
             label="effusion", label_at=(76, 318), label_anchor="ms"),
        dict(shape="polygon", points=[(258, 242), (324, 238), (328, 288), (260, 292)],
             label="effusion", label_at=(292, 314), label_anchor="ms"),
    ]),
    "superimposed-acute-exacerbation-on-fibrosis-hobbs-2021-fig17b.jpg": (2, [
        dict(shape="polygon", points=[(250, 40), (330, 36), (396, 200), (420, 322),
                                      (360, 316), (300, 182), (245, 92)],
             label="honeycombing", label_at=(300, 28), label_anchor="ms"),
        dict(shape="polygon", points=[(120, 430), (250, 400), (390, 430), (400, 620),
                                      (330, 760), (200, 790), (110, 700), (95, 540)],
             label="new ground-glass", label_at=(250, 830), label_anchor="ms"),
    ]),
})
