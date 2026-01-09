import joblib
import inspect

p = joblib.load('models/stage1_pipeline.pkl')
print('Pipeline steps:', list(p.named_steps.keys()))
pre = p.named_steps.get('preprocessor')
print('Preprocessor:', type(pre))

if hasattr(p, 'feature_names_in_'):
    print('pipeline.feature_names_in_ length:', len(p.feature_names_in_))
    print('pipeline.feature_names_in_:', p.feature_names_in_)

if pre is not None:
    try:
        print('preprocessor.transformers_:')
        for t in pre.transformers:
            print('  name:', t[0], 'columns:', t[2], 'transformer type:', type(t[1]))
    except Exception as e:
        print('Could not list transformers:', e)
    try:
        feat = pre.get_feature_names_out()
        print('preprocessor.get_feature_names_out length:', len(feat))
        print('preprocessor.get_feature_names_out:', feat)
    except Exception as e:
        print('get_feature_names_out error:', e)

try:
    model = p.named_steps.get('model')
    print('Model type:', type(model))
except Exception as e:
    print('Model info error:', e)

# Also try to get the number of features the model expects
try:
    bst = p.named_steps['model']
    if hasattr(bst, 'n_features_in_'):
        print('model.n_features_in_:', bst.n_features_in_)
    if hasattr(bst, 'get_booster'):
        booster = bst.get_booster()
        print('Booster feature_names length:', len(booster.feature_names))
        print('Booster feature_names:', booster.feature_names)
except Exception as e:
    print('Model feature info error:', e)

print('Done')
