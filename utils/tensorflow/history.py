import  numpy as np

def saveLoad(file,history):
    print("cnn.history type", type(history))
    hitData = {'accuracy': np.array([]), "val_accuracy": np.array([]), "loss": np.array([]), 'val_loss': np.array([])}
    try:
        loadnpz = np.load(file)
        print("load...", loadnpz['accuracy'])
        hitData["accuracy"] = loadnpz["accuracy"]
        hitData["val_accuracy"] = loadnpz["val_accuracy"]
        hitData["loss"] = loadnpz["loss"]
        hitData['val_loss'] = loadnpz['val_loss']
    except FileNotFoundError:
        print(f"file 없음 :{file} ")


    hits = history.history
    print("11", hits["accuracy"], type(hits["accuracy"]))
    hitData["accuracy"] = np.append(hitData["accuracy"], hits["accuracy"])
    hitData["val_accuracy"] = np.append(hitData["val_accuracy"], hits["val_accuracy"])
    hitData["loss"] = np.append(hitData["loss"], hits["loss"])
    hitData['val_loss'] = np.append(hitData['val_loss'], hits['val_loss'])

    np.savez(file, accuracy=hitData["accuracy"], val_accuracy=hitData["val_accuracy"],
             loss=hitData["loss"], val_loss=hitData["val_loss"])
    return hitData