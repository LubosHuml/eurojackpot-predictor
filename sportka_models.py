import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_sportka_model(window_size=10, num_features=13, lstm_units=64, learning_rate=1e-3):
    """
    Builds the multi-modal LSTM model for Sportka prediction.
    """
    # Inputs
    input_num = layers.Input(shape=(window_size, num_features), name="input_num")
    input_main = layers.Input(shape=(window_size, 6), name="input_main")
    
    # Embedding for main numbers (1 to 49)
    embed_main_layer = layers.Embedding(input_dim=50, output_dim=16, name="embed_main")
    embedded_main = embed_main_layer(input_main) # [batch, w, 6, 16]
    
    # Flatten the numbers' embeddings at each time step
    reshaped_main = layers.Reshape((window_size, 6 * 16))(embedded_main) # [batch, w, 96]
    
    # Concatenate sequence features
    concat_features = layers.Concatenate(axis=-1)([input_num, reshaped_main]) # [batch, w, 13 + 96 = 109]
    
    spatial_dropout = layers.SpatialDropout1D(0.5)(concat_features)
    
    # LSTM layer
    lstm_out = layers.LSTM(
        lstm_units,
        dropout=0.3,
        kernel_regularizer=regularizers.l2(1e-4),
        recurrent_regularizer=regularizers.l2(1e-4),
        name="lstm_core"
    )(spatial_dropout)
    
    # Outputs
    # 1. Sum prediction
    sum_output = layers.Dense(
        1,
        activation="linear",
        kernel_regularizer=regularizers.l2(1e-4),
        name="sum_head"
    )(lstm_out)
    
    # 2. Counts prediction
    counts_output = layers.Dense(
        4,
        activation="linear",
        kernel_regularizer=regularizers.l2(1e-4),
        name="counts_head"
    )(lstm_out)
    
    # 3. Main numbers logits
    main_logits_output = layers.Dense(
        49,
        activation="sigmoid",
        kernel_regularizer=regularizers.l2(1e-4),
        name="main_logits_head"
    )(lstm_out)
    
    # Assemble Model
    model = models.Model(
        inputs=[input_num, input_main],
        outputs=[sum_output, counts_output, main_logits_output],
        name="sportka_model"
    )
    
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss={
            "sum_head": "mse",
            "counts_head": "mse",
            "main_logits_head": "binary_crossentropy"
        },
        loss_weights={
            "sum_head": 1.0,
            "counts_head": 1.0,
            "main_logits_head": 5.0
        }
    )
    return model
