import cv2
import numpy as np

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

def extract_w(u, k1, k2, romega):# renvoie l'image correspondant à la fenêtre de centre (k1,k2)
    M,N = u.shape
    omegax = np.array(range(k1-romega,k1+romega+1))
    omegay = np.array(range(k2-romega,k2+romega+1))
    omegax = omegax % M
    omegay = omegay % M
    uw = u[np.ix_(omegax, omegay)]
    
    return uw

def guided_f(p,I,romega,epsilon):
    M,N = p.shape
    nomega = (1 + 2*romega)**2 # nombre de pixels dans la fenêtre
    a = np.zeros((M,N))
    b = np.zeros((M,N))
    for x in range(M):
        for y in range(N):
            Iw = extract_w(I, x, y, romega)
            pw = extract_w(p, x, y, romega)
            muk = np.mean(Iw)
            sigmak2 = np.var(Iw)
            pbarrek = np.mean(pw)
            a[x,y] = ((1/nomega) * np.sum(Iw*pw) - muk * pbarrek)/(sigmak2 + epsilon)
            b[x,y] = pbarrek - a[x,y] * muk

    q = np.zeros((M,N))
    for x in range(M):
        for y in range(N):
            aw = extract_w(a, x, y, 2*romega)
            bw = extract_w(b, x, y, 2*romega)
            q[x,y] = (1/nomega) * np.sum(aw * p[x,y] + bw)
    
    return q