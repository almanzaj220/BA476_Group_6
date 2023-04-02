LASSO = {'cof': 0.002934049709215787}

REDGE = {'cof': 24.175940791691282}

DECISION_TREE = {'max_depth': 7, 
                 'max_leaf_nodes': 22}

RANDOM_FORIST = {'max_depth': 22 , 
                 'max_leaf_nodes': 300, # The higher the better
                 'n_estimators': 600 } # The higher the better

FEATURE_LIST = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density', 
    'pH',
    'sulphates',
    'alcohol',
    'quality',
    'is_red',
]

