import os

mkpath="../datasets/easy/trainval/"

for i in range(30):
    mkpath_ = mkpath + str(i)
    isExists = os.path.exists(mkpath_)

    if not isExists:
        print(mkpath_)
        os.makedirs(mkpath_)

