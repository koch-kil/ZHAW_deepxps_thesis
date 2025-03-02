from typing import List, Literal, Union
from pathlib import Path
import numpy as np
import pandas as pd


def interpolate_signal(
                        original_indices: np.array,
                        original_signal: np.array,
                        N_points: int = 1024) -> np.array:
    from scipy.interpolate import CubicSpline
    # Define the indices and corresponding values for the original signal
    indices = original_indices
    values = original_signal

    # Create a cubic spline object using the original data
    spline = CubicSpline(indices, values)

    # Generate the indices for the interpolated signal with 1024 points
    interpolated_indices = np.linspace(486.6, 1486.6, N_points)
    

    # Interpolate the signal using the cubic spline
    interpolated_signal = spline(interpolated_indices)

    # interpolated_signal now contains the interpolated signal with 1024 points
    return interpolated_signal


def parse_file(path: Union[str, Path],
               filetype: Literal['vms', 'csv'],
               scale: bool = True,
               N_points: int = 1024,
               energy: Literal['kinetic', 'binding'] = 'kinetic',
               flip: bool = True,
               verbose: bool = False
               ) -> (np.array, np.array, np.array, np.array):
    '''
    Parses a file and returns the x and y values of the original file
    and the processed x and y values

    input:
        path: path to vamas or csv file
        filetype: 'vms' or 'csv'
        scale: scale data to [0,1]
        N_points: number of points of the processed data
        energy: 'kinetic' or 'binding'

    output:
        x: x values of original file
        y: y values of original file
        x_new: processed x values
        y_new: processed y values
    '''
    import base
    # extract label
    
    if isinstance(path, str):
        filename = path.split('\\')[-1]
        if 'multi' in path:
            label = base.pair_list_to_tuples(filename.split('_')[:-1])
        else:
            label = filename.split(',')[0].split('_')[:2]

    elif isinstance(path, Path):
        filename = path.name
        if 'multi' in path:
            label = base.pair_list_to_tuples(filename.split('_')[:-1])
        else:
            label = filename.split(',')[0].split('_')[:2]
            
    if filetype == 'vms':
        from vamas import Vamas
        vamas_data = Vamas(path)
        # check length
        if (np.array([len(i.corresponding_variables[0].y_values) for i in vamas_data.blocks]).max()) < 1000:
            raise ValueError('There is no survey spectra block in the file')
        # check how close to our range
        else:
            valid_blocks = [(n, # enumeration
                        int(i.x_start + len(i.corresponding_variables[0].y_values) * i.x_step)) # end-point of data
                        for n, i in enumerate(vamas_data.blocks)
                        if len(i.corresponding_variables[0].y_values)>900 # more than 900 datapoints
    and abs(int(i.x_start + len(i.corresponding_variables[0].y_values) * i.x_step) - (i.x_label.lower() == 'binding energy' and 20 or 1486)) < 200]
            # finds the blocks with more than 1000 datapoints
            # and within deviation of our range

            if len(valid_blocks) == 0:
                raise ValueError('There is no survey spectra block in the file')
            idx = min(valid_blocks, key=lambda t: t[1])[0]
            if verbose:
                print('Selecting block with {} points'.format(len(vamas_data.blocks[idx].corresponding_variables[0].y_values)))

        data = vamas_data.blocks[idx] # select the block with the most points


        y_len = len(data.corresponding_variables[0].y_values)
        if y_len < 1024:
            raise ValueError('There is no survey spectra block in the file')

        x = np.linspace(start=data.x_start,
                        stop=data.x_start + y_len*(data.x_step),
                        num=y_len,
                        endpoint=False)
        y = np.array(data.corresponding_variables[0].y_values)
        
        if data.x_label.lower() in ['be', 'binding', 'binding energy', 'bindingenergy']:
            energy = 'binding'
        # else:
        #     x = base.kinetic_energy_list_to_binding_energy_al_kalpha(x)
        
        # transform to 1024 points
        # be sure to distinguish kinetic / binding energy scales
        
        if energy == 'kinetic':  # goes from 486 to 1486
            x_new = np.flip(
                        np.linspace(
                            start=max(min(x), 486.6),  # must start at 486.6
                            stop=min(max(x), 1486.6),  # must end at 1486.6
                            num=N_points
                            )
                        )
            # indeces of points inside the range
            inside_indx = np.where((x > 486.6) & (x < 1486.68))
            X_kin = x
            
        elif energy == 'binding':  # goes from 1000 to 0
            x_new = np.flip(
                        np.linspace(
                                    start=max(min(x), 0),  # must start at 0
                                    stop=min(max(x), 1000),  # must end at 1000
                                    num=N_points
                                    )
                            )
            # indeces of points inside the range
            inside_indx = np.where((x > 0) & (x < 1000))
            X_kin = base.binding_energy_list_to_kinetic_energy_al_kalpha(x, 3)

        
        f = len(inside_indx)
        # transform to N_points points
        if f == N_points:
            return x, y, x, y
        
        if f != N_points:
                y_new = interpolate_signal(X_kin, y, N_points=N_points)

        if scale:
            y_new = MaxScaler(y_new)
        if flip:
            y_new = np.flip(y_new)
        
        return x, y, x_new, y_new, label

    elif filetype == 'csv':
        df = pd.read_csv(path,
                         skiprows=7,
                         sep='\t',
                         names=['x', 'y', 'z', 'a'])
        df = df[['x', 'y']]
        f = len(df.x)
        # transform to N_points points
        x_new = np.arange(0, f, step=f/N_points)
        if f == N_points:
            return df.x, df.y, x_new, df.y
        if f > N_points:
            y_new = interpolate_signal(y, N_points=N_points)
        if scale:
            y_new = MaxScaler(y_new)

        return df.x, df.y, x_new, y_new, label

    else:
        raise ValueError('invalid file')


def MaxScaler(x: np.array):
    return (x / max(x))


def MaxScale_df(df: pd.DataFrame):
    return df.apply(MaxScaler)


def extend_data(x: np.array,
                y: np.array,
                desired_range: List[int]):
    '''
    Extends the data to the left and right of the original range.
    It uses numpy polyfit to calculate the slope and intercept of
    the linear function and then extends the data.

    input:
        x: x values of original data
        y: y values of original data
        desired_range: desired range of the data, e.g. [0, 1000]

    output:
        x: extended x values
        y: extended y values

    >>> x = np.array([3, 4, 5, 6, 7, 8])
    >>> y = np.array([0, 1, 2, 3, 4, 5])
    >>> desired_range = [0, 10]
    >>> x_n, y_n = extend_data(x, y, desired_range)
    >>> x_n
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
    >>> y_n
    array([-3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])

    '''

    # extend to the left taking the average of the first 10 points
    if desired_range[0] < x.min():
        n_points = int(x.min() - desired_range[0])
        x_left = np.linspace(desired_range[0], x.min()-1, n_points)
        y_left = np.polyval(np.polyfit(x[:10], y[:10], 1), x_left)
        x = np.concatenate((x_left, x))
        y = np.concatenate((y_left, y))

    # extend to the right taking the average of the last 10 points
    if desired_range[1] > x.max():
        n_points = int(desired_range[1] - x.max())
        x_right = np.linspace(x.max()+1, desired_range[1], n_points)
        y_right = np.polyval(np.polyfit(x[-10:], y[-10:], 1), x_right)
        x = np.concatenate((x, x_right))
        y = np.concatenate((y, y_right))

    return x, y


if __name__ == '__main__':
    import doctest
    doctest.testmod()
