import keras_nlp

preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=128,
)
gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
    "gpt2_base_en", preprocessor=preprocessor
)


tokenizer = keras_nlp.models.GPT2Tokenizer.from_preset("gpt2_base_en")
tokenizer("The quick brown fox jumped.")

# Batched input.
tokenizer(["The quick brown fox jumped.", "The fox slept."])

# Detokenization.
tokenizer.detokenize(tokenizer("The quick brown fox jumped."))

# Custom vocabulary.
vocab = {"<|endoftext|>": 0, "a": 4, "Ġquick": 5, "Ġfox": 6}
merges = ["Ġ q", "u i", "c k", "ui ck", "Ġq uick"]
merges += ["Ġ f", "o x", "Ġf ox"]
tokenizer = keras_nlp.models.GPT2Tokenizer(vocabulary=vocab, merges=merges)
tokenizer("a quick fox.")


start = time.time()

output = gpt2_lm.generate("My trip to Yosemite was", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")


train_ds = train_ds.take(500)
num_epochs = 1

# Linearly decaying learning rate.
learning_rate = keras.optimizers.schedules.PolynomialDecay(
    5e-5,
    decay_steps=train_ds.cardinality() * num_epochs,
    end_learning_rate=0.0,
)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
gpt2_lm.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=loss,
    weighted_metrics=["accuracy"],
)

gpt2_lm.fit(train_ds, epochs=num_epochs)