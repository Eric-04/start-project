from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Lambda, concatenate, add
from keras.models import Model

def create_base_model(input_shape, num_classes):
  
    img_input = Input(shape=input_shape)
    x = Reshape((*input_shape, 1))(img_input)

    # 16 filter convolution
    x = Conv2D(16, 3, activation='elu')(x)

    x = MaxPooling2D((2, 2))(x)

    # 32 filter convolution
    cov_32 = Conv2D(64, 3, activation='elu')
    x = cov_32(x)

    x = MaxPooling2D((2, 2))(x)

    # 64 filter convolution
    cov_64 = Conv2D(128, 3,activation='elu')
    x = cov_64(x)

    final = MaxPooling2D((2, 2))(x)
    
    final = Flatten()(x)#(final)

    # final = Dense(2024, activation='elu')(final)
    # final = BatchNormalization()(final)
    # final = Dropout(0.2)(final)



    final = Dense(64, activation='elu')(final)
    final = BatchNormalization()(final)
    final = Dropout(0.3)(final)



    final = Dense(64, activation='elu')(final)
    final = BatchNormalization()(final)
    final = Dropout(0.3)(final)

    out = Dense(num_classes, activation='softmax')(final)
    model = Model(img_input, out)
    return model