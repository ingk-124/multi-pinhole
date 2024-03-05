# calculate X-ray transparent
import time
import numpy as np
from scipy import interpolate
from pathlib import Path
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from utils.my_stdio import my_print

ua = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ' \
     'AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.6.1 Safari/605.1.15'
options = webdriver.ChromeOptions()
options.add_argument(ua)
options.add_argument('headless')
DATA_PATH = Path().cwd().parent / 'transparent'


def get_data_from_CXRO(material, thickness, save=True, show=False):
    thickness = str(thickness)
    driver = webdriver.Chrome(options=options)
    url = "https://henke.lbl.gov/optical_constants/filter2.html"
    driver.get(url)
    pulldown = driver.find_element(By.NAME, "Material")
    opt = [_.get_attribute('value') for _ in Select(pulldown).options]
    time.sleep(0.5)

    def change_value(name, value):
        driver.execute_script(f'document.getElementsByName("{name}")[0].value = "{value}";')
        return

    if material in opt:
        Select(pulldown).select_by_visible_text(material)
    else:
        change_value("Formula", material)
    change_value("Thickness", thickness)
    change_value("Min", "10")
    change_value("Max", "30000")
    change_value("Npts", "500")
    Select(driver.find_element(By.NAME, "Plot")).select_by_visible_text("LogLin")
    time.sleep(0.5)  # wait for 0.5 sec
    driver.find_element(By.XPATH, "//input[@type='submit' and @value='Submit Request']").click()
    driver.switch_to.window(driver.window_handles[-1])
    time.sleep(0.5)  # wait for 0.5 sec
    try:
        dat_url = driver.find_element(By.XPATH, "//a").get_attribute("href")
    except Exception as e:
        my_print(e, show=show)
        driver.quit()
        return
    response = requests.get(dat_url)
    if save:
        (DATA_PATH / f"{material}").mkdir(exist_ok=True)
        with open(DATA_PATH / f"{material}/{material}_{thickness.replace('.', '_')}.dat", "w") as f:
            f.write(response.text)
    driver.quit()
    return np.array([_.strip().split() for _ in response.text.split('\n') if _][2:], dtype=float)


def get_data(material, thickness, save=True, force=False, show=False):
    thickness = str(thickness)
    file = DATA_PATH / f"{material}/{material}_{thickness.replace('.', '_')}.dat"
    if file.exists() and not force:
        my_print(f"Reading {material} {thickness} um from local...", show=show)
        return np.loadtxt(file, skiprows=2)
    else:
        my_print(f"Downloading {material} {thickness} um from CXRO...", show=show)
        return get_data_from_CXRO(material, thickness, save)


def exp_fit(d1, d2, t1, t2):
    """
    Fit transparent data by exponential
    Parameters
    ----------
    d1: float
        filter1 thickness
    d2: float
        filter2 thickness
    t1: np.ndarray
        filter1 transparent
    t2: np.ndarray
        filter2 transparent

    Returns
    -------
    c: np.ndarray
        coefficient parameters

    Notes
    -----
    The transparency of a thin film is expressed as:
        T = exp(cd),
    where c and d are a coefficient parameters depending on photon energies and thickness of the film respectively.
    """
    c = (np.log(t1 / t2)) / (d1 - d2)
    return np.where(np.isnan(c), 0, c)


def characteristic(material, d, show=False):
    """
    Calculate the transparency of a thin film
    Parameters
    ----------
    material: str
        material name
    d: float
        thickness of the film

    Returns
    -------
    T: np.ndarray
        transparency
    """
    d_1 = d / 10
    d_2 = d / 5
    e_1, t_1 = get_data(material, d_1).T  # eV, T for d/10 thickness
    e_2, t_2 = get_data(material, d_2).T  # eV, T for d/100 thickness
    if np.all(e_1 == e_2):
        my_print("pass", show=show)
        pass
    else:
        e_1, t_1 = get_data(material, d_1, force=True).T  # force update
        e_2, t_2 = get_data(material, d_2, force=True).T  # force update

    c = exp_fit(d_1, d_2, t_1, t_2)

    def transparent(d_, E):
        # convert d_ to array. minimum dimension is 2
        d_ = np.atleast_2d(d_).T
        c_ = interpolate.interp1d(e_1, c, kind="linear")(E)
        return np.exp(c_[None, :] * d_)

    return transparent


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    d = 10
    material = "polyimide"
    film = get_data(material, d)

    plt.plot(film[:, 0], film[:, 1], label=f"{material} {d} um (CXRO)")

    # logspace from 10 to 3e4
    E = np.logspace(1, 4, 1000)
    d_list = [d, d / 0.5, d / 0.2]
    T = characteristic(material, d)(d_list, E)
    for i, d_ in enumerate(d_list):
        plt.plot(E, T[i], label=f"{material} {d_:.2f} um (characteristic)")
    plt.xscale("log")
    plt.legend()
    plt.show()
