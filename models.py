import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_multimodal_model(window_size=10, num_features=13, lstm_units=64, learning_rate=1e-3):
    """
    Builds the multi-modal LSTM model for Eurojackpot prediction.
    
    Parameters:
        window_size (int): w (lookback steps)
        num_features (int): number of statistical/financial features (13)
        lstm_units (int): hidden units for LSTM layer
        learning_rate (float): learning rate for Adam optimizer
        
    Returns:
        tf.keras.Model: compiled multi-output model
    """
    # Inputs
    input_num = layers.Input(shape=(window_size, num_features), name="input_num")
    input_main = layers.Input(shape=(window_size, 5), name="input_main")
    input_euro = layers.Input(shape=(window_size, 2), name="input_euro")
    
    # Embeddings
    # Main numbers: 1 to 50. Input dim = 51 (1 to 50 + index 0 unused), output_dim = 16
    embed_main_layer = layers.Embedding(input_dim=51, output_dim=16, name="embed_main")
    # Euro numbers: 1 to 12. Input dim = 13 (1 to 12 + index 0 unused), output_dim = 8
    embed_euro_layer = layers.Embedding(input_dim=13, output_dim=8, name="embed_euro")
    
    embedded_main = embed_main_layer(input_main) # [batch, w, 5, 16]
    embedded_euro = embed_euro_layer(input_euro) # [batch, w, 2, 8]
    
    # Flatten the numbers' embeddings at each time step (using Reshape to maintain dimensions)
    reshaped_main = layers.Reshape((window_size, 5 * 16))(embedded_main) # [batch, w, 80]
    reshaped_euro = layers.Reshape((window_size, 2 * 8))(embedded_euro) # [batch, w, 16]
    
    # Concatenate sequence features
    concat_features = layers.Concatenate(axis=-1)([input_num, reshaped_main, reshaped_euro]) # [batch, w, 13 + 80 + 16 = 109]
    
    # SpatialDropout1D drops entire channels across the time sequence
    spatial_dropout = layers.SpatialDropout1D(0.5)(concat_features)
    
    # LSTM recurrent layer
    lstm_out = layers.LSTM(
        lstm_units,
        dropout=0.3,
        kernel_regularizer=regularizers.l2(1e-4),
        recurrent_regularizer=regularizers.l2(1e-4),
        name="lstm_core"
    )(spatial_dropout)
    
    # Outputs/Heads
    # 1. Model Sum: regression for next draw sum
    sum_output = layers.Dense(
        1,
        activation="linear",
        kernel_regularizer=regularizers.l2(1e-4),
        name="sum_head"
    )(lstm_out)
    
    # 2. Model Counts: regression for Even, Odd, Low, High counts
    counts_output = layers.Dense(
        4,
        activation="linear",
        kernel_regularizer=regularizers.l2(1e-4),
        name="counts_head"
    )(lstm_out)
    
    # 3. Model Logits: classification representing individual main numbers probabilities
    main_logits_output = layers.Dense(
        50,
        activation="sigmoid",
        kernel_regularizer=regularizers.l2(1e-4),
        name="main_logits_head"
    )(lstm_out)
    
    # 4. Model Logits: classification representing individual Euro numbers probabilities
    euro_logits_output = layers.Dense(
        12,
        activation="sigmoid",
        kernel_regularizer=regularizers.l2(1e-4),
        name="euro_logits_head"
    )(lstm_out)
    
    # Assemble Model
    model = models.Model(
        inputs=[input_num, input_main, input_euro],
        outputs=[sum_output, counts_output, main_logits_output, euro_logits_output],
        name="eurojackpot_model"
    )
    
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss={
            "sum_head": "mse",
            "counts_head": "mse",
            "main_logits_head": "binary_crossentropy",
            "euro_logits_head": "binary_crossentropy"
        },
        loss_weights={
            "sum_head": 1.0,
            "counts_head": 1.0,
            "main_logits_head": 5.0,
            "euro_logits_head": 5.0
        }
    )
    return model
