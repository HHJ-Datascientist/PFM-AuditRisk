def limpieza (df):
    #librerías necesarias
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from imblearn.under_sampling import NearMiss
    from imblearn.over_sampling import SMOTE
    # Eliminamos la variable Id
    df = df.drop("Unnamed: 0", axis = 1) 
    # Establecemos las variables de la X y de la Y. 
    classification_X = df.drop(["misstate"], axis = 1)
    classification_y = df["misstate"]
    # iniciando Smote under-sampling
    sm = SMOTE()
    X_sm, y_sm = sm.fit_resample(classification_X,classification_y)
    nm = NearMiss(version=1)
    X_nm, y_nm = nm.fit_resample(classification_X,classification_y)
    # Definimos el train y el test SMOTE
    X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(X_sm, y_sm, 
                                                                test_size = 0.30, random_state = 0)
    # Definimos el train y el test NEARMISS
    X_train_nm, X_test_nm, y_train_nm, y_test_nm = train_test_split(X_nm, y_nm, 
                                                                test_size = 0.30, random_state = 0)

    return(X_train_nm, X_test_nm, y_train_nm, y_test_nm, X_train_sm, X_test_sm, y_train_sm, y_test_sm)