import os
import numpy as np
import shutil

root_dir = 'workspace/images/'
classes_dir = ['ngfl','ngsc','ngsh', 'ok']

# if not (os.path.exists(root_dir + 'train/') and os.path.exists(root_dir + 'val/') and os.path.exists(root_dir + 'test/')):
for cls in classes_dir:
    if not(os.path.exists(root_dir + 'train2/' + cls)):
        os.makedirs(root_dir + 'train2/' + cls) #path tujuan
    if not(os.path.exists(root_dir + 'val2/' + cls)):
        os.makedirs(root_dir + 'val2/' + cls)
    if not(os.path.exists(root_dir + 'test2/' + cls)):
        os.makedirs(root_dir + 'test2/' + cls)

    source = root_dir + cls #path sumber image
    allFileNames = os.listdir(source) #hitung jumlah image di source
    np.random.shuffle(allFileNames)
    val_ratio = 0.1
    test_ratio = 0.1

    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                        [int(len(allFileNames) * (1 - (val_ratio + test_ratio))),
                                                        int(len(allFileNames) * (1 - test_ratio))])
    train_FileNames = [source+'/'+ name for name in train_FileNames.tolist()]
    val_FileNames = [source+'/'+ name for name in val_FileNames.tolist()]
    test_FileNames = [source+'/' + name for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))
    
    for name in train_FileNames:
        shutil.copy(name, root_dir +'train2/' + cls)
    for name in val_FileNames:
        shutil.copy(name, root_dir +'val2/' + cls)
    for name in test_FileNames:
        shutil.copy(name, root_dir +'test2/' + cls)