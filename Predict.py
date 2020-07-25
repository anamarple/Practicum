

# Predicts second bucket based off of first predicted bucket
# Input: df, column name, vectorizer for e/ sub-bucket, and predictor for e/ sub-bucket
# Returns: list of predicted bucket2 values
def predict_bucket2(df, col_name, vec_l2_comp, vec_l2_geo, vec_l2_prod, vec_l2_gen, pred_l2_comp, pred_l2_geo,
                    pred_l2_prod, pred_l2_gen):
    bucket2_pred = []
    for idx in df.index:
        if str(df[col_name][idx]) == 'Economics':
            pred = 'Economics'
            bucket2_pred.append(pred)

        elif str(df[col_name][idx]) == 'Completion':
            f = vec_l2_comp.transform([df['File'][idx]])
            pred = pred_l2_comp.predict(f)
            bucket2_pred.append(pred[0])

        elif str(df[col_name][idx]) == 'Geological':
            f = vec_l2_geo.transform([df['File'][idx]])
            pred = pred_l2_geo.predict(f)
            bucket2_pred.append(pred[0])

        elif str(df[col_name][idx]) == 'Production':
            f = vec_l2_prod.transform([df['File'][idx]])
            pred = pred_l2_prod.predict(f)
            bucket2_pred.append(pred[0])

        elif str(df[col_name][idx]) == 'General':
            f = vec_l2_gen.transform([df['File'][idx]])
            pred = pred_l2_gen.predict(f)
            bucket2_pred.append(pred[0])
        else:
            print('error')
    return bucket2_pred


# Returns name of sub folder to copy file into based off of assigned Bucket2 name
# Input: Bucket2 category (ex: 'Presentations')
# Returns: Sub folder name (ex: '01 General')
def get_folder(name):

    if name == 'Previous':
        return '00 Previous'

    elif name == 'Presentations' or name == 'General':
        return '01 General'

    elif name == 'Volumetric and Reserves Estimates':
        return '02 Volumetric and Reserves Estimates'

    elif name == 'Production':
        return '03 Production'

    elif name == 'Development Plans':
        return '04 Development Plans'

    elif name == 'Economics':
        return '05 Economics'

    elif name == 'Field Reports':
        return '06 Field Reports'

    elif name == 'Seismic Data':
        return '07 Seismic Data'

    elif name == 'Geologic Maps':
        return '08 Geologic Maps'

    elif name == 'Bubble Maps':
        return '09 Bubble Maps'

    elif name == 'PVT and Test Data':
        return '10 PVT and Test Data'

    elif name == 'Petrophysical Summaries':
        return '11 Petrophysical Summaries'

    elif name == 'Cross-Sections':
        return '12 Cross-Sections'

    elif name == 'Logs':
        return '13 Logs'

    elif name == 'Field Activity':
        return '15 Field Activity'

    elif name == 'Modeling':
        return '16 Modeling'

    return