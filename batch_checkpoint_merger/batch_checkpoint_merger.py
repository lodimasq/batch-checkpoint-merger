import glob
import os
import sys
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import pyperclip


def init_layout():
    interp_list = ['SmoothStep', 'SmootherStep', 'SmoothestStep', 'Exact']
    fp_list = ['Half (FP16)', 'Full (FP32)']
    sg.user_settings_filename(path='..')
    smooth_step_blurb = (
        'Sigmoid-like interpolation model. Smooths the output of the batch by biasing step \n'
        'sizes which are closest to 0 and 1, the area in which merging has the largest effect.\n'
        'The graph below depicts this smoothing effect. Your chosen steps are highlighted.')
    exact_blurb = ('Linear merging model, no interpolation is applied to the step size. (1:1) \n'
                   'The graph below depicts this linear model. Your chosen steps are highlighted.')
    initial_folder = sg.user_settings_get_entry('folder_path', '')
    layout_frame_1 = [
        [sg.In(initial_folder or 'Your Model Folder', size=(32, 20), font='Arial',
               pad=((25, 0), (25, 12)), disabled=True, enable_events=True, k='folder_selected'),
         sg.FolderBrowse(size=(10, 0), font='Arial 10', pad=((9, 25), (25, 12)), k='folder_path')],
        [sg.Text('Select Model A', pad=((25, 0), (25, 0)))],
        [sg.Listbox(list(''), size=(50, 6), pad=((25, 0), (3, 0)), k='model_a')],
        [sg.Text('Select Model B', pad=((25, 0), (25, 0)))],
        [sg.Listbox(list(''), size=(50, 6), pad=((25, 0), (3, 0)), k='model_b')],
        [sg.Text('Output Log', pad=((25, 0), (35, 0)))],
        [sg.Multiline(size=(125, 12), pad=((25, 25), (3, 25)), write_only=True, disabled=True,
                      reroute_stdout=True, reroute_stderr=True, echo_stdout_stderr=True,
                      autoscroll=True, auto_refresh=True)],
        [sg.Ok('Merge', font='Arial 12', pad=((155, 0), (0, 0)), key='merge'),
         sg.Ok('Copy XY', font='Arial 12', key='copy_xy')]
    ]
    input_layout_l = [
        [sg.Text('Start batch from:', font='Arial 12', pad=((20, 0), (25, 0)))],
        [sg.Text('Step Size:', font='Arial 12', pad=((20, 0), (10, 0)))],
        [sg.Text('Nbr. Steps:', font='Arial 12', pad=((20, 0), (10, 0)))]
    ]
    input_layout_r = [
        [sg.In('0.05', size=10, font='Arial 12', pad=((0, 0), (27, 5)), k='batch_start'),
         sg.Text('(0.00 - 1.00) eg. 0.05 = Start at 5% ratio', font='Arial 8 italic',
                 pad=((12, 0), (28, 10)))],
        [sg.In('0.05', size=10, font='Arial 12', pad=((0, 0), (3, 5)), k='step_size'),
         sg.Text('(0.01 - 1.00) eg. 0.05 = Merge in 5% increments', font='Arial 8 italic',
                 pad=((12, 0), (4, 10)))],
        [sg.In('8', size=10, font='Arial 12', pad=((0, 0), (3, 5)), k='nbr_steps'),
         sg.Text('(1 - 100) eg. 8 = 5%,10%,15%,20%,25%,30%,35%,40%', font='Arial 8 italic',
                 pad=((12, 0), (4, 10)))]
    ]
    layout_frame_2 = [
        [sg.Frame('', input_layout_l, size=(150, 135), border_width=0),
         sg.Frame('', input_layout_r, size=(450, 135), border_width=0, pad=(0, 0))],
        [sg.Text('Interpolation Model:', font='Arial 12', pad=((25, 0), (0, 0))),
         sg.Combo(values=interp_list, readonly=True, default_value='SmoothStep',
                  font='Arial 12', pad=((15, 0), (0, 0)), k='interp_model', enable_events=True)],
        [sg.Text('Float Precision:', font='Arial 12', pad=((25, 0), (10, 0))),
         sg.Combo(values=fp_list, readonly=True, default_value='Half (FP16)',
                  font='Arial 12', pad=((41, 0), (10, 0)), k='fp_precision', enable_events=True)],
        [sg.Text(smooth_step_blurb, font='Arial 12', pad=((25, 0), (30, 0)), k='blurb')],
        [sg.Canvas(key='plot_canvas', background_color='#64778d')]
    ]
    layout_window = [[sg.Frame('', layout_frame_1, size=(450, 700), border_width=0),
                      sg.Frame('', layout_frame_2, size=(600, 700), border_width=0)]]
    return exact_blurb, initial_folder, layout_window, smooth_step_blurb


def merge_models(model_a, model_b, file_path, alpha_list, fn_list, interp_model, fp_precision):
    model_0 = torch.load(f'{file_path}/{model_a}')
    model_1 = torch.load(f'{file_path}/{model_b}')
    theta_0 = model_0['state_dict']
    theta_1 = model_1['state_dict']

    for k, alpha in enumerate(alpha_list):
        steps = len(alpha_list)
        filename = f'{model_a[:-5]}_{model_b[:-5]}_{round(fn_list[k], 2)}_{interp_model}.ckpt'
        print(f'({k + 1}/{steps}) {filename}')

        print(f'({k + 1}/{steps}) Merging Common Weights...', end='\r')
        for key in theta_0.keys():
            if 'model' in key and key in theta_1:
                theta_0[key] = (1 - alpha) * theta_0[key] + alpha * theta_1[key]
        print(' Done!')

        print(f'({k + 1}/{steps}) Merging Distinct Weights...', end='\r')
        for key in theta_1.keys():
            if 'model' in key and key not in theta_0:
                theta_0[key] = theta_1[key]
        print(' Done!')

        if fp_precision[-6:] == '(FP16)':
            print(f'({k + 1}/{steps}) Converting to FP16...', end='\r')
            for key in theta_0.keys():
                if 'model' in key and key not in theta_0:
                    theta_0[key] = theta_0[key].to(torch.float16)
            print(' Done!')

        print(f'({k + 1}/{steps}) Saving Model...', end='\r')
        batch_dir = f'{file_path}/~batch_merges'
        Path(batch_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model_a, f'{batch_dir}/{filename}')
        print(' Done!')

    print(' ===============Merge Batch Complete===============')


def copy_xy(model_a, model_b, fn_list, interp_model):
    xy_string = f'{model_a}'
    for i in fn_list:
        filename = f'{model_a[:-5]}_{model_b[:-5]}_{i}_{interp_model}.ckpt'
        xy_string = f'{xy_string}, {filename}'
    xy_string = f'{xy_string}, {model_b}'
    pyperclip.copy(xy_string)


def get_alpha_list(batch_start, interp_model, nbr_steps, step_size):
    batch_end = batch_start + ((nbr_steps - 1) * step_size)
    alpha_range = drange(batch_start, batch_end, step_size)
    alpha_range_0 = drange(batch_start, batch_end, step_size)

    if interp_model == 'SmoothStep':
        return list(alpha_range_0), list(map(smoothstep, alpha_range))
    if interp_model == 'SmootherStep':
        return list(alpha_range_0), list(map(smootherstep, alpha_range))
    if interp_model == 'SmoothestStep':
        return list(alpha_range_0), list(map(smootheststep, alpha_range))
    return list(alpha_range_0), list(alpha_range_0)


def get_filenames(folder):
    path = rf"{folder}/*.ckpt"
    files = list(map(os.path.basename, glob.glob(path)))
    return files


def init_graph(bg_color, fg_color):
    fig = plt.figure(figsize=(6, 4), facecolor=bg_color)
    ax = fig.add_subplot(111)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlim(-0.01, 1.01)
    ax.set_xlabel('Input Ratio', color=fg_color)
    ax.set_ylabel('Interpolated Ratio', color=fg_color)
    for i in ax.spines:
        ax.spines[i].set_color(fg_color)
    ax.tick_params(colors=fg_color)
    ax.grid()
    exact_line_range = list(drange(0, 1.01, 0.01))
    ax.plot(exact_line_range, exact_line_range, 'k-', label='Exact')
    return ax, fig


def plot_lines(ax, batch_start=0.05, interp_model='SmoothStep', nbr_steps=8, step_size=0.05):
    batch_end = batch_start + ((nbr_steps - 1) * step_size)
    demo_range = drange(0, 1.01, 0.01)
    selected_range = drange(batch_start, batch_end, step_size)
    demo_range_org = list(drange(0, 1.01, 0.01))
    selected_range_org = list(drange(batch_start, batch_end, step_size))

    if interp_model == 'SmoothStep':
        demo_range_mod = list(map(smoothstep, demo_range))
        selected_range_mod = list(map(smoothstep, selected_range))
    elif interp_model == 'SmootherStep':
        demo_range_mod = list(map(smootherstep, demo_range))
        selected_range_mod = list(map(smootherstep, selected_range))
    elif interp_model == 'SmoothestStep':
        demo_range_mod = list(map(smootheststep, demo_range))
        selected_range_mod = list(map(smootheststep, selected_range))

    if interp_model == 'Exact':
        selected_line = ax.plot(selected_range_org, selected_range_org, 'ys',
                                label='Selected Steps')
        ax.legend(loc='best')
        return False, selected_line

    demo_line = ax.plot(demo_range_org, demo_range_mod, 'b-', label=interp_model)
    selected_line = ax.plot(selected_range_org, selected_range_mod, 'ys',
                            label='Selected Steps')
    ax.legend(loc='best')
    return demo_line, selected_line


def remove_line(ax, del_line):
    del_line = del_line.pop(0)
    del_line.remove()
    ax.legend(loc='best')


def smoothstep(x):
    return x * x * (3 - 2 * x)


def smootherstep(x):
    return x * x * x * (x * (x * 6 - 15) + 10)


def smootheststep(x):
    y = -20 * pow(x, 7)
    y += 70 * pow(x, 6)
    y -= 84 * pow(x, 5)
    y += 35 * pow(x, 4)
    return y


def drange(start, stop, step):
    value = start
    while value <= stop:
        yield round(value, 2)
        value += step


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def main():
    exact_blurb, initial_folder, layout_window, smooth_step_blurb = init_layout()

    window = sg.Window('Batch Model Merger', layout_window, margins=(0, 0),
                       background_color='#C7D5E0',
                       finalize=True)

    window['batch_start'].bind('<FocusOut>', '_lost')
    window['step_size'].bind('<FocusOut>', '_lost')
    window['nbr_steps'].bind('<FocusOut>', '_lost')

    if initial_folder:
        files = get_filenames(initial_folder)
        window['model_a'].update(files)
        window['model_b'].update(files)

    bg_color = sg.theme_element_background_color()
    fg_color = sg.theme_text_color()

    ax, fig = init_graph(bg_color, fg_color)

    fig_agg = draw_figure(window['plot_canvas'].TKCanvas, fig)
    fig_agg.draw()

    demo_line, selected_line = plot_lines(ax)

    print('Please select your models and options then click "Merge".')

    # This is an Event Loop
    while True:
        event, values = window.read()
        # if event not in (sg.TIMEOUT_EVENT, sg.WIN_CLOSED):
        #     print('============ Event = ', event, ' ==============')
        #     print('-------- Values Dictionary (key=value) --------')
        #     for key in values:
        #         print(key, ' = ', values[key])
        if event in (None, 'Exit'):
            print("Exiting..")
            break
        if event == 'folder_selected':
            files = get_filenames(values['folder_selected'])
            window['model_a'].update(files)
            window['model_b'].update(files)
            sg.user_settings_set_entry('folder_path', values['folder_selected'])
        elif event in ('interp_model', 'batch_start_lost', 'nbr_steps_lost', 'step_size_lost'):
            if values['interp_model'] == 'Exact':
                window['blurb'].update(exact_blurb)
            else:
                window['blurb'].update(smooth_step_blurb)
            if demo_line:
                remove_line(ax, demo_line)
            remove_line(ax, selected_line)
            demo_line, selected_line = plot_lines(ax, float(values['batch_start']),
                                                  values['interp_model'], int(values['nbr_steps']),
                                                  float(values['step_size']))
            fig_agg.get_tk_widget().forget()
            fig_agg = draw_figure(window['plot_canvas'].TKCanvas, fig)
            fig_agg.draw()
        elif event == 'merge':
            print('================Starting Merge Batch================')
            fn_list, alpha_list = get_alpha_list(float(values['batch_start']),
                                                 values['interp_model'],
                                                 int(values['nbr_steps']),
                                                 float(values['step_size']))
            merge_models(values['model_a'][0], values['model_b'][0], values['folder_selected'],
                         alpha_list, fn_list, values['interp_model'], values['fp_precision'])
        elif event == 'copy_xy':
            fn_list, _ = get_alpha_list(float(values['batch_start']),
                                        values['interp_model'],
                                        int(values['nbr_steps']),
                                        float(values['step_size']))
            copy_xy(values['model_a'][0], values['model_b'][0], fn_list, values['interp_model'])

    window.close()
    sys.exit()


if __name__ == '__main__':
    main()
