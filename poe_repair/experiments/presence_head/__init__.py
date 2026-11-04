"""The presence head: two independent "is this animal in the frame" questions.

Captions compete, because each caption's probability is normalised over every caption a picture
could have. Two binary presence questions do not, so a frame holding two animals can score high
on both at once. ``labelling.py`` holds the rule that decides what "both" means.
"""
