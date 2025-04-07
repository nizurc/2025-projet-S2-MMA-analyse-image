import cv2
import numpy as np

def period_im(p): # FONCTION QUI PERIODISE LES IMAGES
    # Créer les symétries
    flip_h = cv2.flip(p, 1)   # Symétrie horizontale
    flip_v = cv2.flip(p, 0)   # Symétrie verticale
    flip_hv = cv2.flip(p, -1) # Symétrie horizontale + verticale

    # Assembler les images pour obtenir une image 4 fois plus grande
    top_row = np.hstack((p, flip_h))   # Ligne du haut
    bottom_row = np.hstack((flip_v, flip_hv))  # Ligne du bas
    p4 = np.vstack((top_row, bottom_row))  # Image complète

    return p4

def deperiod_im(p4): # FONCTION QUI DEPERIODISE LES IMAGES
    M,N = p4.shape
    p = p4[:M//2, :M//2]
    
    return p

def extract_w(u, k1, k2, romega):# renvoie l'image correspondant à la fenêtre de centre (k1,k2)
    M,N = u.shape
    omegax = np.array(range(k1-romega,k1+romega+1))
    omegay = np.array(range(k2-romega,k2+romega+1))
    omegax = omegax % M
    omegay = omegay % M
    uw = u[np.ix_(omegax, omegay)]
    
    return uw

def guided_f(p,I,romega,epsilon): # IMPLEMENTATION NAIVE DU FILTRE GUIDÉ
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
            aw = extract_w(a, x, y, romega)
            bw = extract_w(b, x, y, romega)
            q[x,y] = (1/nomega) * np.sum(aw * p[x,y] + bw)
    
    return q

def compute_intmat(M):# calcule la matrice intégrale
    M1 = np.cumsum(M, axis=1)
    M2 = np.cumsum(M1, axis=0)

    return M2

def compute_averagew(A,p1,p2,q1,q2):# calcule des moyennes dans des fenêtres à partir de la matrice intégrale
    M,N = A.shape
    i = min(max(p1-1, 0), M - 1)
    j = min(max(p2-1, 0), N - 1)
    k = min(max(q1, 0), N - 1)
    l = min(max(q2, 0), N - 1)
    sumw = A[k,l] - A[i,l] - A[k,j] + A[i,j]
    airew = (k-i)*(l-j)
    
    return sumw/airew

def guided_f_fast(p,I,romega,epsilon): # IMPLEMENTATION EFFICACE DU FILTRE GUIDÉ (MATRICE INTÉGRALE)
    M,N = p.shape
    a = np.zeros((M,N))
    b = np.zeros((M,N))
    iInt = compute_intmat(I)
    i2Int = compute_intmat((I-np.mean(I))**2)
    pInt = compute_intmat(p)
    piInt = compute_intmat(p*I)
    for x in range(M):
        for y in range(N):
            muk = compute_averagew(iInt,x-romega,y-romega,x+romega,y+romega)
            sigmak2 = compute_averagew(i2Int,x-romega,y-romega,x+romega,y+romega)
            pbarrek = compute_averagew(pInt,x-romega,y-romega,x+romega,y+romega)
            piw = compute_averagew(piInt,x-romega,y-romega,x+romega,y+romega)
            a[x,y] = (piw - muk * pbarrek)/(sigmak2 + epsilon)
            b[x,y] = pbarrek - a[x,y] * muk

    q = np.zeros((M,N))
    aInt = compute_intmat(a)
    bInt = compute_intmat(b)
    for x in range(M):
        for y in range(N):
            aw = compute_averagew(aInt,x-2*romega,y-2*romega,x+2*romega,y+2*romega)
            bw = compute_averagew(bInt,x-2*romega,y-2*romega,x+2*romega,y+2*romega)
            q[x,y] = p[x,y] * aw + bw 
    
    return q

def average_filter(u,r):
    # uniform filter with a square (2*r+1)x(2*r+1) window 
    # u is a 2d image
    # r is the radius for the filter
   
    (nrow, ncol) = u.shape
    big_uint = np.zeros((nrow+2*r+1,ncol+2*r+1))
    big_uint[r+1:nrow+r+1,r+1:ncol+r+1] = u
    big_uint = np.cumsum(np.cumsum(big_uint,0),1)  # integral image
        
    out = big_uint[2*r+1:nrow+2*r+1,2*r+1:ncol+2*r+1] + big_uint[0:nrow,0:ncol] - big_uint[0:nrow,2*r+1:ncol+2*r+1] - big_uint[2*r+1:nrow+2*r+1,0:ncol]
    out = out/(2*r+1)**2
    
    return out

def guided_f_fast2(p,I,romega,epsilon): # IMPLEMENTATION EFFICACE DU FILTRE GUIDÉ (MATRICE INTÉGRALE)
    M,N = p.shape
    a = np.zeros((M,N))
    b = np.zeros((M,N))

    p_av = f.average_filter(p,romega)
    I_av = f.average_filter(I,romega)
    Ip_av = f.average_filter(I*p,romega)
    sig2_av = f.average_filter((I-I_av)**2,romega)
    a = (Ip_av - p_av*I_av)/(sig2_av + epsilon)
    b = p_av - a * I_av
    a_av = f.average_filter(a,romega)
    b_av = f.average_filter(b,romega)

    q = I * a_av + b_av

    return q