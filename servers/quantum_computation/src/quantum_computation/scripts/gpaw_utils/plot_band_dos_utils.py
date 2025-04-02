import plotly.graph_objects as go
import numpy as np
# from ase.dft.band_structure import BandStructure
from ase.spectrum.band_structure import BandStructure
from gpaw.spinorbit import soc_eigenstates
from ase.dft.dos import DOS
import matplotlib.colors as mcolors


def plot_band(
        calc,
        emin=-6,
        emax=3,
        ylabel='Energy (eV)',
        colors=[
            'blue',
            'red'],
    labels=[
            'w/o SOC',
            'w/ SOC'],
        soc=False,
        only_show_soc=False):
    """
    Plot the band structure with and without SOC, but non spin-polarization
    The band structure with SOC is calculated in a perturbative way.

    Parameters:
    calc (Calculator): The calculator object.
    emin (float): The minimum y-axis value. Default is -6.
    emax (float): The maximum y-axis value. Default is 3.
    ylabel (str): The label for the y-axis. Default is 'Energy (eV)'.
    colors (list[str]): The colors of the band structure. Default is ['blue', 'red'].
    labels (list[str]): The labels of the band structure. Default is ['w/o SOC', 'w/ SOC'].
    soc (bool): Whether to add the band structure with SOC. Default is False.
    only_show_soc (bool): Whether to only show the band structure with SOC. Default is False.
    """

    # the ordinary band structure plot
    bands = calc.band_structure()
    bands = bands.subtract_reference()  # shift the band to Fermi level
    (x, X, xlabels) = bands.path.get_linear_kpoint_axis()
    fig = go.Figure()

    # plot the band structure without SOC
    if not only_show_soc or not soc:
        for i in range(bands.energies.shape[2]):
            fig.add_trace(go.Scatter(
                x=x,
                y=bands.energies[0, :, i],
                mode='lines',
                line=dict(color=colors[0]),
                name=labels[0] if i == 0 else '',
                showlegend=(i == 0)
            ))

    # plot the band structure with SOC
    if soc:
        # calculate the band with SOC in a perturbative way
        soc_calc = soc_eigenstates(calc)
        energies_soc = soc_calc.eigenvalues()
        energies_soc -= soc_calc.fermi_level

        # construct the bandstructure object with SOC
        energies_soc = np.array(energies_soc)
        nkpt = energies_soc.shape[0]
        nband = energies_soc.shape[1]
        energies_soc = energies_soc.reshape(1, nkpt, nband)
        bands_soc = BandStructure(bands.path, energies_soc, reference=0)

        # Plot the SOC band structure
        for i in range(bands_soc.energies.shape[2]):
            fig.add_trace(go.Scatter(
                x=x,
                y=bands_soc.energies[0, :, i],
                mode='lines',
                line=dict(color=colors[1], dash='dash'),
                name=labels[1] if i == 0 else '',
                showlegend=(i == 0)
            ))

    # Update layout
    fig.update_layout(
        xaxis_title='k-points',
        yaxis_title=ylabel,
        yaxis_range=[emin, emax],
        xaxis_tickvals=X,
        xaxis_ticktext=xlabels,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    # Add vertical lines for special k-points
    for x in bands.path.special_points.values():
        fig.add_vline(x=x, line_width=1, line_dash="dash", line_color="gray")

    # save the band structure to json file
    bands.write('bandstructure.json')
    if soc:
        bands_soc.write('bandstructure_soc.json')

    return fig


def plot_band_projection(calc, locfun, scale_factor=100, emin=-6, emax=3,
                         ylabel='Energy (eV)', xlabel='K-path', colors=['blue', 'red'],
                         labels=['Bands', 'Projected Bands']):
    """
    Plot the band structure with projections
    non-spin polarized, no SOC
    """
    bands = calc.band_structure()
    bands = bands.subtract_reference()  # shift the band to Fermi level
    energies = bands.energies
    path = bands.path
    (x, X, xlabels) = path.get_linear_kpoint_axis()

    fig = go.Figure()

    if locfun:
        proj_kni = calc.get_projections(locfun, spin=0)  # Assuming spin-up, change to 1 for spin-down if needed
        projections = np.abs(proj_kni)**2
        projections_sum = np.sum(projections, axis=2)

        # Plot bands without projections as background
        for band in range(energies.shape[2]):
            fig.add_trace(go.Scatter(
                x=x, y=energies[0, :, band],
                mode='lines', line=dict(color=colors[0]),
                name=labels[0] if band == 0 else '',
                showlegend=(band == 0)
            ))

        # Plot the projected bands
        for band in range(energies.shape[2]):
            sizes = projections_sum[:, band] * scale_factor
            fig.add_trace(go.Scatter(
                x=x, y=energies[0, :, band],
                mode='markers',
                marker=dict(size=sizes, color=colors[1], opacity=0.5),
                name=labels[1] if band == 0 else '',
                showlegend=(band == 0)
            ))

        # Add vertical lines for high-symmetry points
        for label_position in X:
            fig.add_vline(x=label_position, line_dash="dash", line_color="gray", line_width=1)

    else:
        raise ValueError('locfun is not provided')
        # # If no projections, just plot regular band structure
        # for band in range(energies.shape[2]):
        #     fig.add_trace(go.Scatter(
        #         x=x, y=energies[0, :, band],
        #         mode='lines', line=dict(color=colors[0]),
        #         name=labels[0] if band == 0 else '',
        #         showlegend=(band == 0)
        #     ))

    # Update layout
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        yaxis_range=[emin, emax],
        xaxis_tickvals=X,
        xaxis_ticktext=xlabels,
        title='Band Structure',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    # Add a horizontal line at Fermi level
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    return fig


def plot_band_spinpol(calc, emin=-3, emax=3, soc=True, only_show_soc=True, show_spinprojection=True,
                      ylabel='Energy (eV)', xlabel='K-path', marker_size=5, marker_style='circle'):
    """
    Plot the band structure with spin polarization,
    with and without SOC
    """
    bands = calc.band_structure()
    efermi = bands.reference
    bands = bands.subtract_reference()  # shift the band to Fermi level

    (x, X, xlabels) = bands.path.get_linear_kpoint_axis()

    fig = go.Figure()

    if not soc:
        only_show_soc = False
        show_spinprojection = False

    if not only_show_soc or not soc:
        # Plot the band structure with spin polarization
        for spin in range(2):  # Assuming spin-polarized calculation
            for band in range(bands.energies.shape[2]):
                fig.add_trace(go.Scatter(
                    x=x, y=bands.energies[spin, :, band],
                    mode='lines',
                    line=dict(color='blue' if spin == 0 else 'red'),
                    name=f'Spin {"up" if spin == 0 else "down"}' if band == 0 else '',
                    showlegend=(band == 0)
                ))

    if soc:
        # Calculate the band with SOC in a perturbative way
        soc_calc = soc_eigenstates(calc)
        energies_soc = soc_calc.eigenvalues().T
        spin_projection = soc_calc.spin_projections()
        energies_soc -= efermi
        s_nk = (spin_projection[:, :, 2].T + 1.0) / 2.0

        if show_spinprojection:
            # Plot the spin projected band structure
            # Plot each band segment separately
            for i in range(len(energies_soc)):
                # Split data at high symmetry points
                split_points = [0] + [np.where(x == pos)[0][0] for pos in X[1:-1]] + [len(x) - 1]

                for j in range(len(split_points) - 1):
                    start_idx = split_points[j]
                    end_idx = split_points[j + 1] + 1

                    segment_x = x[start_idx:end_idx]
                    segment_y = energies_soc[i, start_idx:end_idx]
                    segment_colors = s_nk[i, start_idx:end_idx]

                    fig.add_trace(go.Scatter(
                        x=segment_x,
                        y=segment_y,
                        mode='lines+markers',
                        marker=dict(
                            size=marker_size,
                            color=segment_colors,
                            colorscale='RdBu_r',
                            colorbar=dict(
                                title=dict(
                                    text='Spin projection',
                                    side='right'
                                ),
                                tickmode='array',
                                tickvals=[0, 0.5, 1],
                                ticktext=['Down', 'Mixed', 'Up'],
                                x=1.2,
                                thickness=15,
                                ticklabelposition='outside right'
                            ) if (i == 0 and j == 0) else None,
                            symbol=marker_style,
                            opacity=0.5
                        ),
                        line=dict(color='gray'),
                        name='SOC bands' if (i == 0 and j == 0) else None,
                        showlegend=(i == 0 and j == 0)
                    ))
        else:
            # only plot the SOC band
            for band in range(energies_soc.shape[0]):
                fig.add_trace(go.Scatter(
                    x=x,
                    y=energies_soc[band, :].T,
                    mode='lines',
                    line=dict(color='gray'),
                    name='SOC bands',
                    showlegend=(band == 0)
                ))

    # Update layout
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        yaxis_range=[emin, emax],
        xaxis_tickvals=X,
        xaxis_ticktext=xlabels,
        title='Band Structure with Spin Polarization',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    # Add vertical lines for high-symmetry points
    for label_position in X:
        fig.add_vline(x=label_position, line_dash="dash", line_color="gray", line_width=1)

    # Add a horizontal line at Fermi level
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    return fig


def plot_band_spinpol_projection(calc, locfun, emin=-6, emax=3, colors=['blue', 'red'],
                                 labels=['Spin up', 'Spin down'],
                                 scale_factor=10, xlabel='K-path', ylabel='Energy (eV)',
                                 only_show_projection=False):
    """
    Plot the band structure with projections
    Spin polarized, no SOC
    """
    # sigma =  0.1
    # locfun = [[0, 1, sigma], [0, 0, sigma], [0, 2, sigma], [0, 3, sigma]]
    # colors = ['blue', 'red']
    # labels = ['Spin up', 'Spin down']
    # emin = -6
    # emax = 3
    # ylabel = 'Energy (eV)'
    # xlabel = 'K-path'
    # scale_factor = 10

    bands = calc.band_structure()
    bands = bands.subtract_reference()  # shift the band to Fermi level
    energies = bands.energies
    path = bands.path
    (x, X, xlabels) = path.get_linear_kpoint_axis()

    fig = go.Figure()

    if locfun:
        proj_kni_up = calc.get_projections(locfun, spin=0)  # Assuming spin-up, change to 1 for spin-down if needed
        proj_kni_down = calc.get_projections(locfun, spin=1)
        # sum over all atom projections
        # note: it is possible to split px py pz here
        projections_up = np.sum(np.abs(proj_kni_up)**2, axis=2)
        projections_down = np.sum(np.abs(proj_kni_down)**2, axis=2)

        # Plot the projected bands
        for band in range(energies.shape[2]):
            sizes_up = projections_up[:, band] * scale_factor
            sizes_down = projections_down[:, band] * scale_factor

            # Split data at high symmetry points
            split_points = [0] + [np.where(x == pos)[0][0] for pos in X[1:-1]] + [len(x) - 1]

            for j in range(len(split_points) - 1):
                start_idx = split_points[j]
                end_idx = split_points[j + 1] + 1

                segment_x = x[start_idx:end_idx]
                segment_y_up = energies[0, start_idx:end_idx, band]
                segment_y_down = energies[1, start_idx:end_idx, band]
                segment_sizes_up = sizes_up[start_idx:end_idx]
                segment_sizes_down = sizes_down[start_idx:end_idx]

                fig.add_trace(go.Scatter(
                    x=segment_x,
                    y=segment_y_up,
                    mode='lines+markers',
                    marker=dict(size=segment_sizes_up, color=colors[0], opacity=0.5),
                    line=dict(color=colors[0]),
                    name=labels[0] if (band == 0 and j == 0) else None,
                    showlegend=(band == 0 and j == 0)
                ))
                fig.add_trace(go.Scatter(
                    x=segment_x,
                    y=segment_y_down,
                    mode='lines+markers',
                    marker=dict(size=segment_sizes_down, color=colors[1], opacity=0.5),
                    line=dict(color=colors[1]),
                    name=labels[1] if (band == 0 and j == 0) else None,
                    showlegend=(band == 0 and j == 0)
                ))

        # Add vertical lines for high-symmetry points
        for label_position in X:
            fig.add_vline(x=label_position, line_dash="dash", line_color="gray", line_width=1)
    else:
        raise ValueError('locfun is not provided')

    if only_show_projection == False or locfun is None:
        # If no projections, just pslot regular band structure
        for spin in range(energies.shape[0]):
            for band in range(energies.shape[2]):
                fig.add_trace(go.Scatter(
                    x=x, y=energies[spin, :, band],
                    mode='lines', line=dict(color=colors[spin]),
                    name=labels[spin] if band == 0 else '',
                    showlegend=(band == 0)
                ))

    # Update layout
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        yaxis_range=[emin, emax],
        xaxis_tickvals=X,
        xaxis_ticktext=xlabels,
        title='Band Structure',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    # Add a horizontal line at Fermi level
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    return fig


def plot_dos(calc, emin=-6, emax=3, xlabel='Energy (eV)', ylabel='DOS (states/eV)',
             npts=1000, width=0.1, color='black', label='Total DOS'):
    """
    Extract the total DOS from the GPAW calculator and plot it
    the calculator can be a ground state or a band calculation
    non-spin polarized, no SOC

    Parameters:
    calc (Calculator): The calculator object.
    emin (float): The minimum energy value. Default is -6.
    emax (float): The maximum energy value. Default is 3.
    xlabel (str): The x-axis label. Default is 'Energy (eV)'.
    ylabel (str): The y-axis label. Default is 'DOS (states/eV)'.
    npts (int): Number of points for DOS calculation. Default is 1000.
    width (float): Width parameter for DOS calculation. Default is 0.1.
    color (str): Color for the DOS plot. Default is 'black'.
    label (str): Label for the DOS plot. Default is 'Total DOS'.
    """

    # Calculate total DOS
    dos = DOS(calc, npts=npts, width=width)
    energies = dos.energies
    dos_values = dos.get_dos()

    # Create the plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=energies,
        y=dos_values,
        mode='lines',
        line=dict(color=color),
        name=label
    ))

    # Set axis limits
    mask = (energies >= emin) & (energies <= emax)
    ymax = max(dos_values[mask]) * 1.1  # Add 10% margin for better visibility

    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_range=[emin, emax],
        yaxis_range=[0, ymax],
        title='Density of States',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    # Add a vertical line at Fermi level (E=0)
    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    return fig


def plot_dos_projection(calc, locfun, emin=-6, emax=3, xlabel='Energy (eV)',
                        ylabel='DOS (states/eV)', npts=1000, color='black'):
    """
    Extract the total DOS and PDOS from the GPAW calculator and plot them
    the calculator can be a ground state or a band calculation
    non-spin polarized, no SOC

    Parameters:
    calc (Calculator): The calculator object.
    locfun (list): List of [atom_index, angular, sigma] for PDOS calculation.
    emin (float): The minimum energy value. Default is -6.
    emax (float): The maximum energy value. Default is 3.
    xlabel (str): The x-axis label. Default is 'Energy (eV)'.
    ylabel (str): The y-axis label. Default is 'DOS (states/eV)'.
    npts (int): Number of points for DOS calculation. Default is 1000.
    color (str): Base color for the plots. Default is 'black'.
    """

    if not locfun:
        raise ValueError('locfun is not provided')
        # return plot_dos(calc, emin=emin, emax=emax, xlabel=xlabel, ylabel=ylabel,
        #                 npts=npts, width=0.1, color=color, label='Total DOS')

    width = locfun[0][2]  # read the width from locfun

    # Calculate total DOS
    dos = DOS(calc, npts=npts, width=width)
    energies = dos.energies
    dos_values = dos.get_dos()

    # Calculate PDOS
    pdos_values = []
    for loc in locfun:
        a, l, _ = loc
        angular = 's' if l == 0 else 'p' if l == 1 else 'd' if l == 2 else 'f' if l == 3 else None
        _, pdos = calc.get_orbital_ldos(a=a, angular=angular, npts=npts, width=width)
        pdos_values.append(pdos)

    # Sum up the PDOS
    pdos_sum = sum(pdos_values)

    # Create the plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=energies,
        y=dos_values,
        mode='lines',
        line=dict(color=color),
        name='Total DOS'
    ))

    fig.add_trace(go.Scatter(
        x=energies,
        y=pdos_sum,
        mode='lines',
        line=dict(color=color, dash='dash'),
        name='PDOS'
    ))

    # Set axis limits
    mask = (energies >= emin) & (energies <= emax)
    ymax = max(dos_values[mask]) * 1.1  # Add 10% margin for better visibility

    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_range=[emin, emax],
        yaxis_range=[0, ymax],
        title='Density of States with Projections',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    # Add a vertical line at Fermi level (E=0)
    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    return fig


def plot_dos_spinpol(calc, emin=-6, emax=3, npts=1000, width=0.1, xlabel='Energy (eV)',
                     ylabel='DOS (states/eV)', colors=['red', 'blue'],
                     labels=['Spin up', 'Spin down']):
    """
    Extract the total DOS from the GPAW calculator and plot it
    for spin-polarized calculations

    Parameters:
    calc (Calculator): The calculator object.
    emin (float): The minimum energy value. Default is -6.
    emax (float): The maximum energy value. Default is 3.
    npts (int): Number of points for DOS calculation. Default is 1000.
    width (float): Width parameter for DOS calculation. Default is 0.1.
    xlabel (str): The x-axis label. Default is 'Energy (eV)'.
    ylabel (str): The y-axis label. Default is 'DOS (states/eV)'.
    colors (list): Colors for spin up and down. Default is ['red', 'blue'].
    labels (list): Labels for spin up and down. Default is ['Spin up', 'Spin down'].
    """
    if calc.get_number_of_spins() == 1:
        raise ValueError("The calculation is not spin polarized")
    elif calc.get_number_of_spins() == 2:
        pass
    else:
        raise ValueError(f"Unsupported number of spins: {calc.get_number_of_spins()}. Use 1 or 2.")

    # Convert named colors to hex if necessary
    def to_hex(color):
        if color.startswith('#'):
            return color
        return mcolors.to_hex(color)

    colors = [to_hex(color) for color in colors]

    # Calculate total DOS
    dos = DOS(calc, npts=npts, width=width)
    energies = dos.energies
    dos_up = dos.get_dos(spin=0)
    dos_down = dos.get_dos(spin=1)

    # Create the plot
    fig = go.Figure()

    # Spin up DOS
    fig.add_trace(go.Scatter(
        x=energies,
        y=dos_up,
        fill='tozeroy',
        fillcolor=f'rgba({int(colors[0][1:3], 16)},{int(colors[0][3:5], 16)},{int(colors[0][5:7], 16)},0.5)',
        line=dict(color=colors[0], width=1.5),
        name=labels[0]
    ))

    # Spin down DOS (negated for visual separation)
    fig.add_trace(go.Scatter(
        x=energies,
        y=-dos_down,
        fill='tozeroy',
        fillcolor=f'rgba({int(colors[1][1:3], 16)},{int(colors[1][3:5], 16)},{int(colors[1][5:7], 16)},0.5)',
        line=dict(color=colors[1], width=1.5),
        name=labels[1]
    ))

    # Set axis limits
    mask = (energies >= emin) & (energies <= emax)
    ymax = max(max(dos_up[mask]), max(dos_down[mask])) * 1.1  # Add 10% margin for better visibility

    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_range=[emin, emax],
        yaxis_range=[-ymax, ymax],
        title='Spin-Polarized Density of States',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    # Add a vertical line at Fermi level (E=0)
    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    return fig


def plot_dos_spinpol_projection(calc, locfun, emin=-6, emax=3, npts=1000, xlabel='Energy (eV)',
                                ylabel='DOS (states/eV)', colors=['red', 'blue'],
                                labels=['Spin up', 'Spin down']):
    if not locfun:
        raise ValueError('locfun is not provided')

    width = locfun[0][2]  # read the width from locfun

    if calc.get_number_of_spins() == 1:
        raise ValueError("The calculation is not spin polarized")
    elif calc.get_number_of_spins() == 2:
        pass
    else:
        raise ValueError(f"Unsupported number of spins: {calc.get_number_of_spins()}. Use 1 or 2.")

    # Convert named colors to hex if necessary
    def to_hex(color):
        if color.startswith('#'):
            return color
        return mcolors.to_hex(color)

    colors = [to_hex(color) for color in colors]

    # Calculate total DOS
    dos = DOS(calc, npts=npts, width=width)
    energies = dos.energies
    dos_up = dos.get_dos(spin=0)
    dos_down = dos.get_dos(spin=1)

    # Create the plot
    fig = go.Figure()

    # Spin up DOS
    fig.add_trace(go.Scatter(
        x=energies,
        y=dos_up,
        fill='tozeroy',
        fillcolor=f'rgba({int(colors[0][1:3], 16)},{int(colors[0][3:5], 16)},{int(colors[0][5:7], 16)},0.5)',
        line=dict(color=colors[0], width=1.5),
        name=labels[0]
    ))

    # Spin down DOS (negated for visual separation)
    fig.add_trace(go.Scatter(
        x=energies,
        y=-dos_down,
        fill='tozeroy',
        fillcolor=f'rgba({int(colors[1][1:3], 16)},{int(colors[1][3:5], 16)},{int(colors[1][5:7], 16)},0.5)',
        line=dict(color=colors[1], width=1.5),
        name=labels[1]
    ))

    # Calculate PDOS
    pdos_values_up = []
    pdos_values_down = []
    for loc in locfun:
        a, l, _ = loc
        angular = 's' if l == 0 else 'p' if l == 1 else 'd' if l == 2 else 'f' if l == 3 else None
        _, pdos_up = calc.get_orbital_ldos(a=a, angular=angular, npts=npts, width=width, spin=0)
        _, pdos_down = calc.get_orbital_ldos(a=a, angular=angular, npts=npts, width=width, spin=1)
        pdos_values_up.append(pdos_up)
        pdos_values_down.append(pdos_down)

    # Sum up the PDOS
    pdos_sum_up = sum(pdos_values_up)
    pdos_sum_down = sum(pdos_values_down)

    fig.add_trace(go.Scatter(
        x=energies,
        y=pdos_sum_up,
        mode='lines',
        line=dict(color=colors[0], dash='dash'),
        name=labels[0]
    ))

    fig.add_trace(go.Scatter(
        x=energies,
        y=-pdos_sum_down,
        mode='lines',
        line=dict(color=colors[1], dash='dash'),
        name=labels[1]
    ))

    # Set axis limits
    mask = (energies >= emin) & (energies <= emax)
    ymax = max(dos_up[mask]) * 1.1  # Add 10% margin for better visibility
    ymin = min(-dos_down[mask]) * 1.1

    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_range=[emin, emax],
        yaxis_range=[ymin, ymax],
        title='Density of States with Projections',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    # Add a vertical line at Fermi level (E=0)
    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    # fig.show()
    return fig
