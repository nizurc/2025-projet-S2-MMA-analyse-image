def period_im(p):
    # Créer les symétries
    flip_h = cv2.flip(p, 1)   # Symétrie horizontale
    flip_v = cv2.flip(p, 0)   # Symétrie verticale
    flip_hv = cv2.flip(p, -1) # Symétrie horizontale + verticale

    # Assembler les images pour obtenir une image 4 fois plus grande
    top_row = np.hstack((p, flip_h))   # Ligne du haut
    bottom_row = np.hstack((flip_v, flip_hv))  # Ligne du bas
    p4 = np.vstack((top_row, bottom_row))  # Image complète

    return p4