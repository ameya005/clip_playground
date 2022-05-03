"""
Gradio launcher for CLIP playground
"""

import os
import numpy as np
from pathlib import Path

import torch
import clip
import kmapper as km
from sklearn import cluster
import gradio as gr
from PIL import Image
from kmapper.plotlyviz import plotlyviz
from matplotlib import pyplot as plt

def greet(name):
    """
    Greet function
    """
    return "Hello, " + name + "!"


prompts = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


def predict(img_enc, labels):
    logits = img_enc @ labels.T
    return logits


def CLIP_predictor(img, text_labels, image_enc_ch, analysis_type):
    """
    Loads clip model, images and text labels, state
    """
    labels = text_labels.split(',')
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    model, preprocess = clip.load(image_enc_ch)
    img = preprocess(img).unsqueeze(0)
    img = img.cuda()
    img_enc = model.encode_image(img)
    img_enc /= img_enc.norm(dim=-1, keepdim=True)
    zeroshot_wts = []
    mapper = km.KeplerMapper(verbose=False)
    if analysis_type == 'Prompts':
        for label in labels:
            texts = [prompt.format(label) for prompt in prompts]
            label_text = clip.tokenize(texts).cuda()
            class_embeddings = model.encode_text(label_text)
            class_embeddings /= class_embeddings.norm(dim=1, keepdim=True)
            zeroshot_wts.append(class_embeddings)
        zeroshot_wts = torch.stack(zeroshot_wts, dim=2).cuda()
        prompt_size, emb_size, num_labels = zeroshot_wts.shape
        zeroshot_wts_emb = zeroshot_wts.permute(0,2,1).reshape(prompt_size*num_labels, -1)
        logits = predict(img_enc, zeroshot_wts_emb)
        
        pred_logits = logits.reshape(prompt_size, num_labels)
        pred_probs, pred_indices = torch.topk(pred_logits, 2, dim=1)
        margin = pred_probs[:,0] - pred_probs[:,1]
        print(margin.shape)
        pred = pred_logits.mean(dim=0).detach().cpu().numpy().argmax()
        print('logit shape', logits.shape)

        G = mapper.map(margin.cpu().detach().numpy(),
               zeroshot_wts[:,:,pred].cpu().detach().numpy(),
               cover = km.Cover(n_cubes=10,
                               perc_overlap=0.5),
               clusterer=cluster.AgglomerativeClustering(n_clusters=2, affinity='cosine', linkage='complete')
               )
        result_html = mapper.visualize(G,
                    #  lens=pred_probs[:,0].cpu().detach().numpy(),
                    #  lens_names=["Margin"],
                     #custom_tooltips=imagenet_templates,
                     #X_names=imagenet_templates,
                    #  color_values=col_vals,
                    #  color_function_name=['Absolute error'],
                     #title="Confidence Graph for a MLP trained on MNIST",
                     path_html="results/result2.html")
        fig = plotlyviz(G, title='Prompt dashboard', filename='results/prompt_dashboard.png', graph_data=True)
        res_img = Image.open('results/prompt_dashboard.png')
        #fig = plt.figure(figsize=(16,12))
        #km.draw_matplotlib(G, fig=fig)
    elif analysis_type == 'Synonyms':
        pass
    elif analysis_type == 'Negation':
        pass

    return labels[pred.item()], 'results/result.html', res_img#str(logits.detach().cpu().numpy().tolist())


def CLIP_kmap_graph():
    """
    Loads clip model, returns k-map graph
    """
    pass


clip_choices = clip.available_models()
analysis_choices = ['Prompts', 'Synonyms', 'Negation']

demo = gr.Interface(CLIP_predictor, ["image", "text", gr.inputs.Radio(
    clip_choices, default=clip_choices[0], label='Image Encoder'),
    gr.inputs.Radio(analysis_choices, default=analysis_choices[0], label='Label Perturbations')], 
    outputs=['label', 'file', 'image' ], allow_flagging='never',
    examples=[['examples/celeba_test.png', 'male,female', 'RN50', 'Prompts']])
#gr.outputs.Plot(type='matplotlib')
if __name__ == "__main__":
    if not os.path.exists('results'):
        os.mkdir('results')
    demo.launch()
