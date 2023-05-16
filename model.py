from keras.models import Model, Sequential
from keras.layers import Dense, Input,Conv2D,MaxPool2D,GlobalAveragePooling2D,LSTM,Flatten,Input
from keras.layers import TimeDistributed, LSTM, Dense, Dropout,Activation,GlobalMaxPool2D,BatchNormalization

#importing necessary libraries
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization,LSTM,Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint



def CNN_LSTM(shape):

    img_input = Input(shape=(40,40,1))
    input1=Conv2D(32,1)(img_input)
    input2=Flatten()(input1)

    m1=Model(img_input, input2)

    # then create our final model
    model1 =Sequential()
    model1.add(TimeDistributed(m1, input_shape=shape))

    model1.add(LSTM(64,return_sequences=True))
    x0=Flatten()(model1.output)
   
    x=Dense(128,activation='relu')(x0)
    x=Dense(64,activation='relu')(x)
    x1=Dense(16,activation='softmax')(x)

    model = Model(inputs=model1.input, outputs=x1)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                    metrics=['accuracy'])
    print(model.summary())
    return model  



def MY_CNN_LSTM(shape): 
    model = Sequential() # define CNN model 
    model.add(TimeDistributed(Conv2D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(40,40,1))))
    model.add(TimeDistributed(MaxPooling2D(pool_size=5, strides = 2, padding = 'same'))) 
    model.add(TimeDistributed(Flatten())) # define LSTM model 
    model.add(LSTM(128, return_sequences=True))
    model.add(Dense(units=3, activation='softmax')) 
    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

    #model summary
    print(model.summary())
    return model

def CNN_LSTM_(x_train):
    model=Sequential()
    model.add(Conv2D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(40,40,1)))
    model.add(MaxPooling2D(pool_size=5, strides = 2, padding = 'same'))

    model.add(Conv2D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=5, strides = 2, padding = 'same'))

    model.add(Conv2D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=5, strides = 2, padding = 'same'))
    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(LSTM(128, return_sequences=True))

    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
  

    model.add(Dense(units=3, activation='softmax'))

    #compiling the model
    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

    #model summary
    print(model.summary())
    return model

if __name__=='__main__':
    
    CNN_LSTM((114,40,40,1))



