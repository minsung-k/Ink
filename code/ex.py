from util import model_0503 as model

import sys

# ---------------------------------------------------------------------------------

data_num = int(sys.argv[1])
weight = float(sys.argv[2])
epochs = int(sys.argv[3])


# ------------------------------------------------------------------------


model_ = model.model()

#model.full_training(data_num,model_, epochs, weight)
model.full_half_training(data_num,model_, epochs, weight)
