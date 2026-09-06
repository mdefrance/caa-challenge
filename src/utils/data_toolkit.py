import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from warnings import warn


def get_unq_col(data: pd.DataFrame) -> pd.DataFrame:
    """groups columns by its unique names"""
    col_groups = {unq_name(col): [] for col in data}
    for col in data:
        col_groups[unq_name(col)] += [col]
    return col_groups


def unq_name(col):
    """returns the unique name of a column"""
    unq = "".join(c for c in col if not c.isdigit())
    return (
        unq.replace("_MMAX_A", "")
        .replace("_MSOM_A", "")
        .replace("_MM_A", "")
        .replace("AB_", "_")
        .replace("MAX_", "_")
    )


departement_to_region = {
    "01": "Auvergne-Rhône-Alpes",
    "02": "Hauts-de-France",
    "03": "Auvergne-Rhône-Alpes",
    "04": "Provence-Alpes-Côte d'Azur",
    "05": "Provence-Alpes-Côte d'Azur",
    "06": "Provence-Alpes-Côte d'Azur",
    "07": "Auvergne-Rhône-Alpes",
    "08": "Grand Est",
    "09": "Occitanie",
    "10": "Grand Est",
    "11": "Occitanie",
    "12": "Occitanie",
    "13": "Provence-Alpes-Côte d'Azur",
    "14": "Normandie",
    "15": "Auvergne-Rhône-Alpes",
    "16": "Nouvelle-Aquitaine",
    "17": "Nouvelle-Aquitaine",
    "18": "Centre-Val de Loire",
    "19": "Nouvelle-Aquitaine",
    "21": "Bourgogne-Franche-Comté",
    "22": "Bretagne",
    "23": "Nouvelle-Aquitaine",
    "24": "Nouvelle-Aquitaine",
    "25": "Bourgogne-Franche-Comté",
    "26": "Auvergne-Rhône-Alpes",
    "27": "Normandie",
    "28": "Centre-Val de Loire",
    "29": "Bretagne",
    "2A": "Corse",
    "2B": "Corse",
    "20": "Corse",
    "30": "Occitanie",
    "31": "Occitanie",
    "32": "Occitanie",
    "33": "Nouvelle-Aquitaine",
    "34": "Occitanie",
    "35": "Bretagne",
    "36": "Centre-Val de Loire",
    "37": "Centre-Val de Loire",
    "38": "Auvergne-Rhône-Alpes",
    "39": "Bourgogne-Franche-Comté",
    "40": "Nouvelle-Aquitaine",
    "41": "Centre-Val de Loire",
    "42": "Auvergne-Rhône-Alpes",
    "43": "Auvergne-Rhône-Alpes",
    "44": "Pays de la Loire",
    "45": "Centre-Val de Loire",
    "46": "Occitanie",
    "47": "Nouvelle-Aquitaine",
    "48": "Occitanie",
    "49": "Pays de la Loire",
    "50": "Normandie",
    "51": "Grand Est",
    "52": "Grand Est",
    "53": "Pays de la Loire",
    "54": "Grand Est",
    "55": "Grand Est",
    "56": "Bretagne",
    "57": "Grand Est",
    "58": "Bourgogne-Franche-Comté",
    "59": "Hauts-de-France",
    "60": "Hauts-de-France",
    "61": "Normandie",
    "62": "Hauts-de-France",
    "63": "Auvergne-Rhône-Alpes",
    "64": "Nouvelle-Aquitaine",
    "65": "Occitanie",
    "66": "Occitanie",
    "67": "Grand Est",
    "68": "Grand Est",
    "69": "Auvergne-Rhône-Alpes",
    "70": "Bourgogne-Franche-Comté",
    "71": "Bourgogne-Franche-Comté",
    "72": "Pays de la Loire",
    "73": "Auvergne-Rhône-Alpes",
    "74": "Auvergne-Rhône-Alpes",
    "75": "Île-de-France",
    "76": "Normandie",
    "77": "Île-de-France",
    "78": "Île-de-France",
    "79": "Nouvelle-Aquitaine",
    "80": "Hauts-de-France",
    "81": "Occitanie",
    "82": "Occitanie",
    "83": "Provence-Alpes-Côte d'Azur",
    "84": "Provence-Alpes-Côte d'Azur",
    "85": "Pays de la Loire",
    "86": "Nouvelle-Aquitaine",
    "87": "Nouvelle-Aquitaine",
    "88": "Grand Est",
    "89": "Bourgogne-Franche-Comté",
    "90": "Bourgogne-Franche-Comté",
    "91": "Île-de-France",
    "92": "Île-de-France",
    "93": "Île-de-France",
    "94": "Île-de-France",
    "95": "Île-de-France",
    "971": "Guadeloupe",
    "972": "Martinique",
    "973": "Guyane",
    "974": "La Réunion",
    "976": "Mayotte",
}


def get_region(departement_code):
    found = departement_to_region.get(
        str(int(departement_code)).zfill(2), f"Unknown {departement_code}"
    )
    if "Unknown" in found:
        warn(f"Unknown departement code: {departement_code}")
    return found


class Processor(BaseEstimator, TransformerMixin):
    """Feature engineering for both pipelines.

    `data_dir` is where the auxiliary `Incendies.csv` is read from. Leave it None and it
    falls back to the module-level DATA_DIR, which is the repository's own `data/` unless
    CAA_DATA_DIR overrides it -- so nothing changes for the notebooks. Pass it explicitly
    where the file lives somewhere else and setting an environment variable before the
    import would be awkward:

        proc = Processor(data_dir="/kaggle/input/caa-challenge-data")

    It is stored unresolved, per scikit-learn's rule that __init__ only records its
    arguments; `resolve_data_dir()` does the fallback at call time.
    """

    def __init__(self, data_dir: str | Path | None = None):
        self.log_means = None
        self.log_vetuste_means = None
        self.men_means = None
        self.ind_means = None
        self.ind_snv_means = None
        self.altitude_means = None
        self.data_dir = data_dir

    def resolve_data_dir(self) -> Path:
        """The directory to read auxiliary data from: the argument, else DATA_DIR."""
        return Path(self.data_dir) if self.data_dir is not None else DATA_DIR

    def fit(self, data):
        x = data.copy()
        x = format_zone(x)
        x = format_log(x)
        x = format_men(x)
        x = format_ind(x)
        x = format_altitude(x)

        self.log_means = x.groupby(["ZONE_REGION"])["LOG_TOT"].mean().sort_values()
        self.log_vetuste_means = (
            x.groupby(["ZONE_REGION"])["LOG_VETUSTE"].mean().sort_values()
        )
        self.men_means = x.groupby(["ZONE_REGION"])["MEN_TOT"].mean().sort_values()
        self.ind_means = x.groupby(["ZONE_REGION"])["IND_TOT"].mean().sort_values()
        self.ind_snv_means = (
            x.groupby(["ZONE_REGION"])["IND_SNV_num"].mean().sort_values()
        )
        self.altitude_means = (
            x.groupby(["ZONE_REGION"])["ALTITUDE_TOT"].mean().sort_values()
        )
        return self

    def transform(self, data):

        # getting region
        data = format_zone(data)

        # CA ratios
        data = format_ca(data)

        # HOUSING TYPES
        data = format_log(data)
        data["LOG_REGION"] = data["LOG_TOT"].divide(
            data["ZONE_REGION"].map(self.log_means)
        )
        data["LOG_VETUSTE_REGION"] = data["LOG_VETUSTE"].divide(
            data["ZONE_REGION"].map(self.log_vetuste_means)
        )
        # menages
        data = format_men(data)
        data["MEN_REGION"] = data["MEN_TOT"].divide(
            data["ZONE_REGION"].map(self.men_means)
        )

        # IND
        data = format_ind(data)
        data["IND_REGION"] = data["IND_TOT"].divide(
            data["ZONE_REGION"].map(self.ind_means)
        )
        data["IND_SNV_REGION"] = data["IND_SNV_num"].divide(
            data["ZONE_REGION"].map(self.ind_means)
        )

        # ALTITUDE
        data = format_altitude(data)
        data["ALTITUDE_REGION"] = data["ALTITUDE_TOT"].divide(
            data["ZONE_REGION"].map(self.altitude_means)
        )

        # crossing temperature data
        data = cross_temp_data(data)

        # one hot encoding
        data = one_hot_encode(data)

        # adding fire data
        data = add_incendies_info(data, data_dir=self.resolve_data_dir())
        data["VENT_x_CASERNES"] = (
            data["ZONE_VENT"].astype(str) + "__" + data["NB_CASERNES"]
        )

        # kapital data
        kapital_cols = [
            c
            for c in data.columns
            if c.startswith("KAPITAL") and data[c].dtype != "object"
        ]
        data["KAPITAL_SUM"] = data[kapital_cols].sum(axis=1)
        data["KAPITAL_MAX"] = data[kapital_cols].max(axis=1)

        # ohe for derogations
        data = format_derog(data)

        return data


def one_hot_encode(data: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "DEROG1",
        "DEROG6",
        "DEROG7",
        "DEROG9",
        "DEROG10",
        "DEROG11",
        "KAPITAL36",
        "KAPITAL38",
        "KAPITAL39",
    ]:
        data[col] = (data[col] == "O").astype(int)
    return data


def cross_temp_data(data: pd.DataFrame) -> pd.DataFrame:

    temp_columns = [col for col in data.columns if "_MMAX_A" in col or "_MM_A" in col]
    for col1 in temp_columns:
        for col2 in temp_columns:
            if col1 != col2:
                if (col1.endswith("_MM_A") and col2.endswith("_MMAX_A")) or (
                    col1.endswith("_MM_A_Y") and col2.endswith("_MMAX_A_Y")
                ):
                    if col1.replace("_MM_A", "") == col2.replace("_MMAX_A", ""):
                        data[col1 + "_" + col2] = data[col1] + "_" + data[col2]
                        # data.drop([col1, col2], axis=1, inplace=True)
    return data


def format_zone(data: pd.DataFrame) -> pd.DataFrame:
    # only ~100 unique zones: resolve regions once per unique value, then map
    region_map = {z: get_region(z) for z in data["ZONE"].unique()}
    data["ZONE_REGION"] = data["ZONE"].map(region_map)
    data["ZONE"] = data["ZONE"].astype(str).str.zfill(2)
    return data


def format_ca(data: pd.DataFrame) -> pd.DataFrame:

    # CA ratios
    data[["CA1", "CA2", "CA3"]] = data[["CA1", "CA2", "CA3"]].replace(0, np.nan)
    data["CA_TOT"] = data[["CA1", "CA2", "CA3"]].sum(axis=1).replace(0, np.nan)
    data["CA_MEAN"] = data[["CA1", "CA2", "CA3"]].mean(axis=1).replace(0, np.nan)
    for col in ["CA1", "CA2", "CA3"]:
        data[col + "_CA_TOT"] = data[col].divide(data["CA_TOT"])
        data[col + "_CA_MEAN"] = data[col].divide(data["CA_MEAN"])

    return data


def format_men(data: pd.DataFrame) -> pd.DataFrame:

    # converting to numerical
    cols = [
        "MEN_PAUV",
        "MEN_1IND",
        "MEN_5IND",
        "MEN_PROP",
        "MEN_FMP",
        "MEN_COLL",
        "MEN_MAIS",
        "MEN",
    ]
    data = categorical_conversion(data, cols)

    # total number of menage
    data["MEN_TOT"] = (
        data["MEN_PAUV_num"]
        + data["MEN_1IND_num"]
        + data["MEN_5IND_num"]
        + data["MEN_PROP_num"]
        + data["MEN_FMP_num"]
        + data["MEN_COLL_num"]
        + data["MEN_MAIS_num"]
    )
    for col in [
        "MEN_PAUV",
        "MEN_1IND",
        "MEN_5IND",
        "MEN_PROP",
        "MEN_FMP",
        "MEN_COLL",
        "MEN_MAIS",
        "MEN",
    ]:
        data[col + "_MEN_TOT"] = data[col + "_num"].divide(data["MEN_TOT"])
        if col != "MEN":
            data[col + "_MEN"] = data[col + "_num"].divide(data["MEN_num"])
    return data


def format_log(data: pd.DataFrame) -> pd.DataFrame:
    cols = ["LOG_AVA1", "LOG_A1_A2", "LOG_A2_A3", "LOG_APA3", "LOG_INC", "LOG_SOC"]
    data = categorical_conversion(data, cols)

    # total number of housing
    data["LOG_TOT"] = (
        data["LOG_AVA1_num"]
        + data["LOG_A1_A2_num"]
        + data["LOG_A2_A3_num"]
        + data["LOG_APA3_num"]
        + data["LOG_INC_num"]
    )

    # proportion of each type of housing
    for col in [
        "LOG_AVA1_num",
        "LOG_A1_A2_num",
        "LOG_A2_A3_num",
        "LOG_APA3_num",
        "LOG_INC_num",
        "LOG_SOC_num",
    ]:
        data[col + "_LOG_TOT"] = data[col].divide(data["LOG_TOT"])

    # vetusty of buildings
    data["LOG_VETUSTE"] = (
        data["LOG_AVA1_num"] * 80
        + data["LOG_A1_A2_num"] * 50
        + data["LOG_A2_A3_num"] * 30
        + data["LOG_APA3_num"] * 10
    ).divide(data["LOG_TOT"])
    return data


def format_ind(data: pd.DataFrame) -> pd.DataFrame:

    # converting to num
    cols = [
        "IND",
        "IND_0_Y1",
        "IND_Y1_Y2",
        "IND_Y2_Y3",
        "IND_Y3_Y4",
        "IND_Y4_Y5",
        "IND_Y5_Y6",
        "IND_Y6_Y7",
        "IND_Y7_Y8",
        "IND_Y8_Y9",
        "IND_Y9",
        "IND_INC",
        "IND_SNV",
    ]
    data = categorical_conversion(data, cols)

    # total number of ind
    data["IND_TOT"] = (
        data["IND_0_Y1_num"]
        + data["IND_Y1_Y2_num"]
        + data["IND_Y2_Y3_num"]
        + data["IND_Y3_Y4_num"]
        + data["IND_Y4_Y5_num"]
        + data["IND_Y5_Y6_num"]
        + data["IND_Y6_Y7_num"]
        + data["IND_Y7_Y8_num"]
        + data["IND_Y8_Y9_num"]
        + data["IND_Y9_num"]
        + data["IND_INC_num"]
    )
    for col in [
        "IND",
        "IND_0_Y1",
        "IND_Y1_Y2",
        "IND_Y2_Y3",
        "IND_Y3_Y4",
        "IND_Y4_Y5",
        "IND_Y5_Y6",
        "IND_Y6_Y7",
        "IND_Y7_Y8",
        "IND_Y8_Y9",
        "IND_Y9",
        "IND_INC",
        "IND_SNV",
    ]:
        data[col + "_IND_TOT"] = data[col + "_num"].divide(data["IND_TOT"])
        if col != "IND":
            data[col + "_IND"] = data[col + "_num"].divide(data["IND_num"])
        if col != "IND_SNV":
            data[col + "_IND_SNV"] = data[col + "_num"].divide(data["IND_SNV_num"])

    return data


def format_altitude(data: pd.DataFrame) -> pd.DataFrame:

    cols = ["ALTITUDE_1", "ALTITUDE_2", "ALTITUDE_3", "ALTITUDE_4", "ALTITUDE_5"]
    data = categorical_conversion(data, cols)

    data["ALTITUDE_TOT"] = (
        data["ALTITUDE_1_num"]
        + data["ALTITUDE_2_num"]
        + data["ALTITUDE_3_num"]
        + data["ALTITUDE_4_num"]
        + data["ALTITUDE_5_num"]
    )
    for col in ["ALTITUDE_1", "ALTITUDE_2", "ALTITUDE_3", "ALTITUDE_4", "ALTITUDE_5"]:
        data[col + "_ALT_TOT"] = data[col + "_num"].divide(data["ALTITUDE_TOT"])

    return data


categorical_converter = {
    "01. <= 10": 1,
    "02. <= 20": 10,
    "03. <= 30": 20,
    "04. <= 40": 30,
    "05. <= 50": 40,
    "06. <= 60": 50,
    "07. <= 70": 60,
    "08. <= 80": 70,
    "09. <= 90": 80,
    "10. > 90": 90,
    #
    "01. <= 17204": 17204,
    "02. <= 153098": 153098,
    "03. <= 670263": 670263,
    #
    "01. <= 21873": 21873,
    "02. <= 24733": 24733,
    "03. <= 29681": 29681,
    "04. >= 29681": 29681,
    #
    "01. <= 33995": 33995,
    "02. <= 318560": 318560,
    "03. <= 1340814": 1340814,
    # ALTITUDE
    "01. <= 190": 1,
    "02. <= 438": 190,
    "03. <= 794": 438,
    "04. >= 794": 794,
    "01. <= 239": 1,
    "02. <= 588": 239,
    "03. <= 1186": 588,
    "04. >= 1186": 1186,
    "01. <= 236": 1,
    "02. <= 579": 236,
    "03. <= 1178": 579,
    "04. >= 1178": 1178,
    "01. <= 143": 1,
    "02. <= 333": 143,
    "03. <= 645": 333,
    "04. >= 645": 645,
    "01. <= 337": 1,
    "02. <= 840": 337,
    "03. <= 1810": 840,
    "04. >= 1810": 1810,
}


def categorical_conversion(data: pd.DataFrame, cols: list[str]):
    """converts values of categorical features to numerical"""
    # .map is far faster than .replace; keep original value where unmatched (replace semantics)
    new = {}
    for c in cols:
        mapped = data[c].map(categorical_converter)
        new[c + "_num"] = mapped.where(mapped.notna(), data[c])
    return pd.concat([data, pd.DataFrame(new, index=data.index)], axis=1)


# https://bdiff.agriculture.gouv.fr/indicateurs/cartes
fire_extinction_rates = {
    "Aucun feu": [],
    # White: >85% of fires controlled before 1 ha
    ">85%": [
        13,
        83,
        84,
        30,
        40,
        19,
        63,
        43,
        73,
        74,
        68,
        88,
        21,
        18,
        85,
        49,
        72,
        35,
        29,
        14,
        76,
        60,
        8,
        59,
    ],
    # Yellow: 70-85% of fires controlled before 1 ha
    "70-85%": [
        20,
        11,
        66,
        4,
        6,
        5,
        26,
        38,
        47,
        33,
        24,
        16,
        17,
        87,
        86,
        56,
        41,
        45,
        89,
        77,
    ],
    # Orange: 50-70% of fires controlled before 1 ha
    "50-70%": [64, 34, 48, 7, 69, 71, 58, 23, 37, 44, 22, 27, 91, 57, 67],
    # Red: <50% of fires controlled before 1 ha
    "<50%": [9, 31, 81, 12, 32, 82, 46, 15, 36, 53, 28, 10, 52, 54],
}
fire_extinction_rates = {
    str(value).zfill(2): key
    for key, values in fire_extinction_rates.items()
    for value in values
}

total_surface_2023 = {
    "Aucun feu": [50, 61, 78, 95, 75, 80, 62, 2, 51, 55, 70, 25, 39, 1, 3, 42, 79, 65],
    # White: >85% of fires controlled before 1 ha
    "<10ha": [],
    # Yellow: 70-85% of fires controlled before 1 ha
    "10-20ha": [27, 28, 89, 21, 17, 15, 46, 47, 32, 81, 38, 84],
    # Orange: 50-70% of fires controlled before 1 ha
    "20-50ha": [26, 9, 24, 41, 56, 52, 88],
    # Orange: 50-70% of fires controlled before 1 ha
    "50-100ha": [83, 12, 33, 40, 91],
    "100-200ha": [4, 5, 30, 48, 64, 82, 54],
    # Red: <50% of fires controlled before 1 ha
    ">200ha": [6, 20, 13, 7, 34, 11, 66],
}
total_surface_2023 = {
    str(value).zfill(2): key
    for key, values in total_surface_2023.items()
    for value in values
}


total_surface_5y = {
    "Aucun feu": [22, 56, 44, 78, 95, 75, 51, 59],
    # White: >85% of fires controlled before 1 ha
    "<10ha": [
        29,
        50,
        14,
        61,
        53,
        76,
        80,
        62,
        60,
        2,
        8,
        91,
        77,
        10,
        54,
        57,
        67,
        88,
        68,
        70,
        25,
        90,
        74,
        73,
        71,
        69,
        36,
        85,
        79,
    ],
    # Yellow: 70-85% of fires controlled before 1 ha
    "10-20ha": [35, 37, 18, 23, 42, 39],
    # Orange: 50-70% of fires controlled before 1 ha
    "20-50ha": [
        81,
        82,
        32,
        47,
        16,
        87,
        19,
        63,
        43,
        3,
        58,
        89,
        21,
        52,
        1,
        38,
        5,
        49,
        72,
        28,
    ],
    # Orange: 50-70% of fires controlled before 1 ha
    "50-100ha": [17, 86, 41, 24, 12, 26, 45],
    "100-200ha": [64, 65, 31, 9, 46, 15, 48],
    # Red: <50% of fires controlled before 1 ha
    ">200ha": [],
}
total_surface_5y = {
    str(value).zfill(2): key
    for key, values in total_surface_5y.items()
    for value in values
}


surface_over_forest = {
    "Absence de feu": [50, 61, 80, 62, 2, 51, 55, 70, 25, 39, 1, 3, 42, 79, 65],
    "<0.05": [],
    "0.05-0.1": [54, 48, 7, 5, 6],
    "0.1-0.2": [82, 11, 34, 13, 20],
    "0.5-2": [66],
}
surface_over_forest = {
    str(value).zfill(2): key
    for key, values in surface_over_forest.items()
    for value in values
}

# data directory, resolved relative to this file so any checkout works; override with the
# CAA_DATA_DIR environment variable. See data/README.md for the downloads.
DATA_DIR = Path(
    os.environ.get("CAA_DATA_DIR", Path(__file__).resolve().parents[2] / "data")
)


def get_incendies_natures(data_dir: str | Path | None = None) -> pd.DataFrame:
    """fire-incident counts per zone, loaded on first use rather than at import.

    The fallback is applied here rather than inside the cached function, so that None and
    an explicit DATA_DIR are one cache entry instead of two loads of the same file.
    """
    directory = Path(data_dir) if data_dir is not None else DATA_DIR
    return _incendies_natures(directory)


@lru_cache(maxsize=4)
def _incendies_natures(directory: Path) -> pd.DataFrame:
    """The uncached-once-per-directory body. Keyed on the resolved path."""

    # getting data about fires
    incendies = pd.read_csv(directory / "Incendies.csv", sep=",")

    # formatting zone
    incendies["zone"] = incendies["Département"].apply(
        lambda u: str(u).zfill(2).replace("2A", "20").replace("2B", "20")
    )

    # getting count of event per zone
    incendies_natures = incendies.groupby(["zone"]).apply(
        lambda u: (
            pd.DataFrame(u.Nature.fillna("Unknown").value_counts())
            .to_dict()
            .get("count")
        ),
        include_groups=False,
    )

    # creating table for joining
    new_incendies_natures = []
    for zone in incendies_natures.reset_index().to_dict(orient="records"):
        if len(zone.get("zone")) < 3:
            new_incendies_natures += [{**zone.get(0), "zone": zone.get("zone")}]
    return pd.DataFrame(new_incendies_natures)


def add_incendies_info(
    data: pd.DataFrame, data_dir: str | Path | None = None
) -> pd.DataFrame:
    """adds incendies info to the dataset; data_dir defaults to DATA_DIR"""

    data["total_surface_2023"] = data["ZONE"].map(total_surface_2023).fillna("<10ha")
    data["total_surface_5y"] = data["ZONE"].map(total_surface_5y).fillna(">200ha")
    data["surface_over_forest"] = data["ZONE"].map(surface_over_forest).fillna("<0.05")
    data["fire_extinction_rates"] = (
        data["ZONE"].map(fire_extinction_rates).fillna("Aucun feu")
    )
    data["total_surface_crossed"] = (
        data["total_surface_2023"] + "__" + data["total_surface_5y"]
    )
    data["nb_casernes_extinction_rate"] = (
        data["NB_CASERNES"] + "__" + data["fire_extinction_rates"]
    )
    data["zone_vent_extinction_rate"] = (
        data["ZONE_VENT"].astype(str) + "__" + data["fire_extinction_rates"]
    )

    # adding to dataset
    data = data.join(
        get_incendies_natures(data_dir).fillna(0).set_index("zone"), on="ZONE"
    )

    return data


def format_derog(data: pd.DataFrame) -> pd.DataFrame:
    """formats derogation columns to be more usable for ML models"""
    data["DEROG13_formatted"] = (
        data["DEROG13"].map({"D12": 1, "D18": 2}).fillna(0).astype(int)
    )
    data["DEROG8_formatted"] = (data["DEROG8"] == "O").astype(int)
    data["DEROG3_formatted"] = (data["DEROG3"] == "O").astype(int)
    data["DEROG16_formatted"] = (data["DEROG16"] == "SA").astype(int)
    data["DEROG14_formatted"] = (data["DEROG14"] == "D14").astype(int)
    return data


def collapse_count(data: np.ndarray) -> np.ndarray:
    """collapses count values 2 when they are greater than 1, to avoid overfitting on rare events"""
    return data.where(data <= 1, 2)
