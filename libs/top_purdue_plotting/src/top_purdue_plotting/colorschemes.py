main_blue = "#5790fc"
main_orange = "#f89c20"
main_red = "#e42536"
main_maroon = "#964a8b"
main_gray = "#9c9ca1"
main_purple = "#7a21dd"

alt_blue = "#3f90da"
alt_yellow = "#ffa90e"
alt_red = "#bd1f01"
alt_gray = "#94a4a2"
alt_purple = "#832db6"
alt_brown = "#a96b59"
alt_orange = "#e76300"
alt_tan = "#b9ac70"
alt_darkgray = "#717581"
alt_cyan = "#92dadd"

reconstruction_method_colors = {
    "transformer": main_blue,
    "nu2flows": main_maroon,
    "hybrid": main_purple,
    "mlp": main_orange,
    "mlb_weighting": main_red,

    # All terms for "the ground truth"
    "gen": main_gray,
    "true": main_gray,
    "truth": main_gray,
    "target": main_gray,
}

process_colors = {
    "data": "black",

    "ttbar": alt_red,
    
    "ttbarbg": alt_orange,

    "singletop": alt_cyan,
    "tw": alt_cyan,

    "etat": alt_purple,
    "toponium": alt_purple,
    
    "dy": alt_blue,
    "zjets": alt_blue,

    "ttw": alt_yellow,
    "ttz": alt_yellow,
    "ttv": alt_yellow,

    "ww": alt_gray,
    "wz": alt_gray,
    "zz": alt_gray,
    "vv": alt_gray,

    "wjets": alt_darkgray,
    
    "zjets": alt_brown,
}
