
from kotogram.analysis import FormalityLevel, GenderLevel, RegisterLevel

# Number of classes for each task
NUM_FORMALITY_CLASSES = 6
NUM_GRAMMATICALITY_CLASSES = 2  # grammatic (1) vs agrammatic (0)
NUM_GENDER_PRAGMATIC_CLASSES = 2 # pragmatic (1) vs unpragmatic (0)
NUM_REGISTER_CLASSES = 14

# Label mappings
FORMALITY_LABEL_TO_ID = {
    FormalityLevel.VERY_FORMAL: 0,
    FormalityLevel.FORMAL: 1,
    FormalityLevel.NEUTRAL: 2,
    FormalityLevel.CASUAL: 3,
    FormalityLevel.VERY_CASUAL: 4,
    FormalityLevel.UNPRAGMATIC_FORMALITY: 5,
}
FORMALITY_ID_TO_LABEL = {v: k for k, v in FORMALITY_LABEL_TO_ID.items()}

GENDER_LABEL_TO_ID = {
    GenderLevel.MASCULINE: 0,
    GenderLevel.FEMININE: 1,
    GenderLevel.NEUTRAL: 2,
    GenderLevel.UNPRAGMATIC_GENDER: 3,
}
GENDER_ID_TO_LABEL = {v: k for k, v in GENDER_LABEL_TO_ID.items()}

REGISTER_LABEL_TO_ID = {
    RegisterLevel.NEUTRAL: 0,
    RegisterLevel.SONKEIGO: 1,
    RegisterLevel.KENJOGO: 2,
    RegisterLevel.KANSAIBEN: 3,
    RegisterLevel.HAKATABEN: 4,
    RegisterLevel.KYOSHIGO: 5,
    RegisterLevel.NETSLANG: 6,
    RegisterLevel.OJOUSAMA: 7,
    RegisterLevel.GUNTAI: 8,
    RegisterLevel.JOSEIGO: 9,
    RegisterLevel.DANSEIGO: 10,
    RegisterLevel.BURIKKO: 11,
    RegisterLevel.TOHOKU: 12,
    RegisterLevel.BUSHI: 13,
}
REGISTER_ID_TO_LABEL = {
    0: RegisterLevel.NEUTRAL,
    1: RegisterLevel.SONKEIGO,
    2: RegisterLevel.KENJOGO,
    3: RegisterLevel.KANSAIBEN,
    4: RegisterLevel.HAKATABEN,
    5: RegisterLevel.KYOSHIGO,
    6: RegisterLevel.NETSLANG,
    7: RegisterLevel.OJOUSAMA,
    8: RegisterLevel.GUNTAI,
    9: RegisterLevel.JOSEIGO,
    10: RegisterLevel.DANSEIGO,
    11: RegisterLevel.BURIKKO,
    12: RegisterLevel.TOHOKU,
    13: RegisterLevel.BUSHI,
}
