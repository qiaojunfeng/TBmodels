#!/usr/bin/env python
import matplotlib.pyplot as plt
import tbmodels as tb

if __name__ == "__main__":
    BUILD_DIR = "./build_write_hmn"

    # model = tb.Model.from_wannier_folder(BUILD_DIR, prefix="silicon")
    model = tb.Model.from_wannier_tb_file(
        tb_file=f'{BUILD_DIR}/silicon_tb.dat',
        wsvec_file=f'{BUILD_DIR}/silicon_wsvec.dat'
    )

    energy, dos = model.dos([20, 20, 20])
    # print(energy, dos)

    plt.plot(energy, dos)
    plt.savefig('dos.png')
    # plt.show()
