"""
Camera-free core shared by training, the live CLI and (later) the API:

    preprocessing  landmarks -> normalized points -> skeleton image
    model          SkeletonCNN
    inference      checkpoint loading + single-frame prediction
    temporal       per-frame predictions -> committed letters
"""
