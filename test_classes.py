from src.data.TokenDataObj import TokenDataObj
from tensorflow.keras.preprocessing.text import Tokenizer
from src.models.Transformer.SingleEmbeddedTransformer import SingleEmbeddedTransformer
from src.models.Transformer.Transformer import Transformer

path_to_text_file = "./data/processed/linetext.txt"
input_token = Tokenizer()

sequence_length = 80
batch_size = 64
buffer_size = 10000
embedding_dimension = 256
dense_dimension = 256
num_heads = 2
num_att_layers = 2
dropout_rate = 0.1

def text_to_array(input_path: str) -> list:
    file = open(input_path, 'r')
    return file.readlines()#[:10000]


def test_token_data_obj():
    test_text = TokenDataObj(tokenizer=input_token)
    test_text.iniatiate_with_loader(text_to_array, input_path = path_to_text_file)
    test_text.final_dataset(sequence_length=sequence_length, batch_size=batch_size, buffer_size=buffer_size,  padding_required=True)
    

def test_single_emb_trans():
    test_text = TokenDataObj(tokenizer=input_token)
    test_text.iniatiate_with_loader(text_to_array, input_path = path_to_text_file)
    test_text.final_dataset(sequence_length=sequence_length, batch_size=batch_size, buffer_size=buffer_size,  padding_required=True)

    single_emb_trans = SingleEmbeddedTransformer(vocab_size=test_text.vocab_size+1,embedding_dimension=embedding_dimension
                                                ,sequence_length=sequence_length
                                                ,num_heads=num_heads,dense_dimension=dense_dimension,num_att_layers=num_att_layers)

def test_single_trans():
    test_text = TokenDataObj(tokenizer=input_token)
    test_text.iniatiate_with_loader(text_to_array, input_path = path_to_text_file)
    test_text.final_dataset(sequence_length=sequence_length, batch_size=batch_size, buffer_size=buffer_size,  padding_required=True)

    single_emb_trans = Transformer(vocab_size=test_text.vocab_size+1,context_vocab_size=test_text.vocab_size+1,embedding_dimension=embedding_dimension
                                                ,sequence_length=sequence_length
                                                ,num_heads=num_heads,dense_dimension=dense_dimension,num_att_layers=num_att_layers)


def test_classes():
    test_token_data_obj()
    test_single_emb_trans()
    test_single_trans()
    print("All tests passed")

test_classes()
