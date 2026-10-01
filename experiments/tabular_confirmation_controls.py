"""Four frozen candidates per conventional control; no implicit test selection."""
def candidates(family, seed):
    if family == 'trees':
        from sklearn.ensemble import HistGradientBoostingClassifier
        return [(str(steps),HistGradientBoostingClassifier(max_iter=steps,
            max_leaf_nodes=15,min_samples_leaf=5,learning_rate=.1,
            early_stopping=False,random_state=seed)) for steps in (16,32,64,128)]
    if family == 'logistic':
        from sklearn.linear_model import LogisticRegression
        return [(str(c),LogisticRegression(C=c,max_iter=1000,random_state=seed))
                for c in (.01,.1,1.,10.)]
    if family == 'catboost':
        import catboost
        if catboost.__version__ != '1.2.10':
            raise ValueError('Install the pinned confirmation requirements')
        return [(f'd{depth}_l2{l2}',catboost.CatBoostClassifier(iterations=256,
            depth=depth,l2_leaf_reg=l2,learning_rate=.05,loss_function='Logloss',
            random_seed=seed,thread_count=1,task_type='CPU',verbose=False,
            use_best_model=False,allow_writing_files=False))
                for depth,l2 in ((3,1.),(3,5.),(6,1.),(6,5.))]
    raise ValueError('Unknown control family')
