import torch, os
import torch.nn as nn
from torch.autograd import Variable
from data_loading import *
from rdkit import Chem
from rdkit.Chem import rdDepictor, AllChem, Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
'''
the model
'''
class generative_model(nn.Module):
    def __init__(self, vocabs_size, hidden_size, output_size, embedding_dimension, n_layers):
        super(generative_model, self).__init__()
        self.vocabs_size = vocabs_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_dimension = embedding_dimension
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocabs_size, embedding_dimension)
        self.rnn = nn.LSTM(embedding_dimension, hidden_size, n_layers, dropout = 0.2)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        batch_size = input.size(0)
        input = self.embedding(input)
        output, hidden = self.rnn(input.view(1, batch_size, -1), hidden)
        output = self.linear(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        hidden=(Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return hidden

data,vocabs=load_data()
data = set(list(data))
vocabs = list(vocabs)
vocabs_size = len(vocabs)
output_size = len(vocabs)
batch_size = 128
cpu = True
hidden_size = 1024
embedding_dimension = 248
n_layers=3
end_token = ' '

model = generative_model(vocabs_size,hidden_size,output_size,embedding_dimension,n_layers)
model.load_state_dict(torch.load('mytraining.pt', map_location=torch.device('cpu')))
if cpu:
    model = model.cpu()
model.eval()

def checking_mol(smile):
    file = open('data/chebi_smiles.txt', 'r')
    i = 0
    for line in file:
        i+=1
        line = line.strip('%s\n')
        if line == smile:
            message = 'This molecule already exists. Check line {}'.format(i)
            break
        else:
            message = 'This is a new generated molecule'
    file.close()
    return message

def common_template(smile):
    template = Chem.MolFromSmiles(smile)
    AllChem.Compute2DCoords(template)
    ms = [Chem.MolFromSmiles(smi) for smi in mol_batch]
    for m in ms:
        AllChem.GenerateDepictionMatching2DStructure(m,template)
        
    img=Draw.MolsToGridImage(ms,molsPerRow=4,subImgSize=(200,200),legends=[x.GetProp("_Name") for x in ms])
    return img.save('molgrid.png')

def print_mol(smile):
    mol = Chem.MolFromSmiles(smile)
    return Chem.Draw.MolToImage(mol)

def evaluate(prime_str='!', temperature=0.4):
    max_length = 200
    inp = Variable(tensor_from_chars_list(prime_str,vocabs,cpu)).cpu()
    batch_size = inp.size(0)
    hidden = model.init_hidden(batch_size)
    if cpu:
        hidden = (hidden[0].cpu(), hidden[1].cpu())
    predicted = prime_str
    while True:
        output, hidden = model(inp, hidden)
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        # Add predicted character to string and use as next input
        predicted_char = vocabs[top_i]

        if predicted_char ==end_token or len(predicted)>max_length:
            return predicted

        predicted += predicted_char
        inp = Variable(tensor_from_chars_list(predicted_char,vocabs,cpu)).cpu()
        #print(predicted)
    return predicted

def valid_smile(smile):
    return Chem.MolFromSmiles(smile) is not None

def get_canonical_smile(smile):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smile))

def valid_smiles_at_temp(temp = 0.4):
    range_test = 100
    c=0
    for i in range(range_test):
        s= evaluate(prime_str='!', temperature=temp)[1:] # remove the first character !.
        if valid_smile(s):
            print(s)
            c+=1
    return 'percentage of valid molecules in the test batch : {}'.format(float(c/range_test))

def smiles_in_db(smile):
    smile = '!'+get_canonical_smile(smile)+' '
    if smile in data:
        return True
    return False

def percentage_variety_of_valid_at_temp(temp = 0.4):
    range_test = 100
    c_v=0
    c_nd=0
    for i in range(range_test):
        s= evaluate(prime_str='!', temperature=temp)[1:] # remove the first character !.
        if valid_smile(s):
            c_v+=1
            if not smiles_in_db(s):
                c_nd+=1
    return 'Rate of new generated molecules in the test batch : {}'.format(float(c_nd/c_v))
