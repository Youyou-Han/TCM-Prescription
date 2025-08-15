import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book','prescription','presF']:
    dataset = dataloader.Loader_prescription(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN,
    'lightgcn_llm': model.LightGCN_llm,
    'lightgcn_semantic': model.LightGCN_semantic,  # 使用语义增强版本
    'lightgcn_cooccur': model.LightGCN_Cooccur,
    'lightgcn_symptom' : model.LightGCN_Symptom
}