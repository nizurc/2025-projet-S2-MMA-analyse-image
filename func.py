import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def average_filter2(u, r):
    """
    Applique un filtre moyen sur une image 2D u avec une fenêtre carrée (2r+1)x(2r+1),
    en utilisant une image intégrale pour une implémentation rapide.
    Les bords sont symétrisés (padding miroir).
    """
    
    # Padding symétrique (miroir)
    padded = np.pad(u, ((r+1, r), (r+1, r)), mode='symmetric')
    
    # Calcul de l'image intégrale (somme cumulée 2D)
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    
    # Zone centrale : du coin supérieur gauche au coin inférieur droit
    A = integral[2*r+1:,2*r+1:]
    B = integral[:-2*r-1,2*r+1:]
    C = integral[2*r+1:,:-2*r-1]
    D = integral[:-2*r-1,:-2*r-1]
    
    total = A - B - C + D
    out = total / ((2*r + 1) ** 2)
    
    return out

def guided_f_fast2(p,I,romega,epsilon): # IMPLEMENTATION EFFICACE DU FILTRE GUIDÉ (MATRICE INTÉGRALE)
    M,N = p.shape
    a = np.zeros((M,N))
    b = np.zeros((M,N))

    p_av = average_filter2(p,romega)
    I_av = average_filter2(I,romega)
    Ip_av = average_filter2(I*p,romega)
    sig2_av = average_filter2((I-I_av)**2,romega)
    a = (Ip_av - p_av*I_av)/(sig2_av + epsilon)
    b = p_av - a * I_av
    a_av = average_filter2(a,romega)
    b_av = average_filter2(b,romega)

    q = I * a_av + b_av

    return q

def guided_f_fast3(p, I, romega, epsilon): # IMPLEMENTATION EFFICACE DU FILTRE GUIDÉ (MATRICE INTÉGRALE) POUR IMAGE EN COULEUR
    M = p.shape[0] 
    N = p.shape[1]

    # Si l'image I est en couleur, elle doit avoir 3 canaux
    if len(I.shape) == 3 and I.shape[2] == 3:
        # Initialiser la sortie pour une image en couleur
        q = np.zeros_like(I)

        # Appliquer le filtre pour chaque canal de couleur (R, G, B)
        for c in range(3):
            p_channel = p[:, :, c]
            I_channel = I[:, :, c]
            
            a = np.zeros((M, N))
            b = np.zeros((M, N))

            p_av = average_filter2(p_channel, romega)
            I_av = average_filter2(I_channel, romega)
            Ip_av = average_filter2(I_channel * p_channel, romega)
            sig2_av = average_filter2((I_channel - I_av)**2, romega)
            a = (Ip_av - p_av * I_av) / (sig2_av + epsilon)
            b = p_av - a * I_av
            a_av = average_filter2(a, romega)
            b_av = average_filter2(b, romega)
            q[:, :, c] = np.clip(I_channel * a_av + b_av, 0, 1)  # Assurer que les valeurs sont dans la plage [0, 1]

    else:
        # Si l'image n'est pas en couleur, on applique la fonction comme avant
        a = np.zeros((M, N))
        b = np.zeros((M, N))

        p_av = average_filter2(p, romega)
        I_av = average_filter2(I, romega)
        Ip_av = average_filter2(I * p, romega)
        sig2_av = average_filter2((I - I_av)**2, romega)
        a = (Ip_av - p_av * I_av) / (sig2_av + epsilon)
        b = p_av - a * I_av
        a_av = average_filter2(a, romega)
        b_av = average_filter2(b, romega)

        q = I * a_av + b_av

    return q

def compare_parameters_fg(p1,ranger,rangeeps,xmin,xmax,prof):
    # Création d'une grille pour afficher les images
    fig, axes1 = plt.subplots(nrows=len(ranger), ncols=len(rangeeps), figsize=(12, 12))
    axes1 = axes1.flatten()

    if prof ==1:
        # Création d'une grille pour afficher les profils
        fig, axes2 = plt.subplots(nrows=len(ranger), ncols=len(rangeeps), figsize=(12, 12))
        axes2 = axes2.flatten()

    # Compteur d'image
    i = 0
    for r in ranger:
        for epsilon in rangeeps**2:
            img = guided_f_fast3(p1, p1, r, epsilon)
            ax = axes1[i]
            ax.imshow(img, cmap='gray')
            ax.set_title(f"r={r}, ε={epsilon:.3f}")
            ax.axis('off')
            if prof ==1:
                # Calcul et affichage des profils moyens
                prof1 = np.mean(img, axis=0)
                prof2 = np.mean(p1, axis=0)
                ax2 = axes2[i]
                ax2.plot(prof2, label='o', linewidth=1)
                ax2.plot(prof1, label='fg', linewidth=1)
                ax2.set_title(f"r={r}, ε={epsilon:.3f}")
                ax2.legend()
                ax2.grid(True)  # Grille
                ax2.set_ylim(0, 1)  # Uniformisation des axes Y
                ax2.set_xlim(xmin, xmax)

            i += 1

    plt.tight_layout()
    plt.show()

def compare_parameters_fg_zoom(p1,ranger,rangeeps,xmin,xmax,ymin,ymax):
    
    # Création d'une grille pour afficher les images
    fig, axes1 = plt.subplots(nrows=len(ranger), ncols=len(rangeeps), figsize=(12, 12))
    axes1 = axes1.flatten()

    # Compteur d'image
    i = 0
    for r in ranger:
        for epsilon in rangeeps**2:
            img = guided_f_fast3(p1, p1, r, epsilon)
            ax = axes1[i]
            ax.imshow(img, cmap='gray')
            ax.set_title(f"r={r}, ε={epsilon:.3f}")
            ax.axis('off')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymax, ymin)

            i += 1

    plt.tight_layout()
    plt.show()



