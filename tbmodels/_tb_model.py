# -*- coding: utf-8 -*-
#
# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
# pylint: disable=too-many-lines,invalid-name
"""
Implements the Model class, which describes a tight-binding model.
"""

import re
import os
import copy
import time
import itertools
import contextlib
import typing as ty
import collections as co

import h5py
import numpy as np
import scipy.linalg as la
from scipy.special import factorial
import symmetry_representation as sr
from fsc.export import export
from fsc.hdf5_io import subscribe_hdf5, HDF5Enabled

from . import _check_compatibility
from . import _sparse_matrix as sp
from .kdotp import KdotpModel


@export
@subscribe_hdf5('tbmodels.model', check_on_load=False)
class Model(HDF5Enabled):
    """
    A class describing a tight-binding model. It contains methods for modifying the model, evaluating the Hamiltonian or eigenvalues at specific k-points, and writing to and from different file formats.

    Parameters
    ----------
    on_site :
        On-site energy of the states. This is equivalent to having a
        hopping within the same state and the same unit cell (diagonal
        terms of the R=(0, 0, 0) hopping matrix). The length of the list
        must be the same as the number of states.
    hop :
        Hopping matrices, as a dict containing the corresponding lattice
        vector R as a key.
    size :
        Number of states. Defaults to the size of the hopping matrices,
        if such are given.
    dim :
        Dimension of the tight-binding model. By default, the dimension
        is guessed from the other parameters if possible.
    occ :
        Number of occupied states.
    pos :
        Positions of the orbitals, in reduced coordinates. By default,
        all orbitals are set to be at the origin, i.e. at [0., 0., 0.].
    uc :
        Unit cell of the system. The unit cell vectors are given as rows
        in a ``dim`` x ``dim`` array
    contains_cc :
        Specifies whether the hopping matrices and on-site energies are
        given fully (``contains_cc=True``), or the complex conjugate
        should be added for each term to obtain the full model. The
        ``on_site`` parameter is not affected by this.
    sparse :
        Specifies whether the hopping matrices should be saved in sparse
        format.
    """
    def __init__(
        self,
        *,
        on_site: ty.Optional[ty.Collection[float]] = None,
        hop: ty.Optional[ty.Mapping[ty.Tuple[int, ...], ty.Any]] = None,
        size: ty.Optional[int] = None,
        dim: ty.Optional[int] = None,
        occ: ty.Optional[int] = None,
        pos: ty.Optional[ty.Collection[ty.Collection[float]]] = None,
        uc: ty.Optional[np.ndarray] = None,
        contains_cc: bool = True,
        sparse: bool = False
    ):
        if hop is None:
            hop = dict()

        # ---- SPARSITY ----
        self._sparse: bool
        self._matrix_type: ty.Callable[..., ty.Any]
        self.set_sparse(sparse)

        # ---- SIZE ----
        self._init_size(size=size, on_site=on_site, hop=hop, pos=pos)

        # ---- DIMENSION ----
        self._init_dim(dim=dim, hop=hop, pos=pos)

        # ---- UNIT CELL ----
        self.uc = None if uc is None else np.array(uc)  # implicit copy

        # ---- HOPPING TERMS AND POSITIONS ----
        self._init_hop_pos(on_site=on_site, hop=hop, pos=pos, contains_cc=contains_cc)

        # ---- CONSISTENCY CHECK FOR SIZE ----
        self._check_size_hop()

        # ---- CONSISTENCY CHECK FOR DIM ----
        self._check_dim()

        # ---- OCCUPATION NR ----
        self.occ = None if (occ is None) else int(occ)

    #---------------- INIT HELPER FUNCTIONS --------------------------------#
    def _init_size(self, size, on_site, hop, pos):
        """
        Sets the size of the system (number of orbitals).
        """
        if size is not None:
            self.size = size
        elif on_site is not None:
            self.size = len(on_site)
        elif pos is not None:
            self.size = len(pos)
        elif hop:
            self.size = next(iter(hop.values())).shape[0]
        else:
            raise ValueError(
                'Empty hoppings dictionary supplied and no size, on-site energies or positions given. Cannot determine the size of the system.'
            )

    def _init_dim(self, dim, hop, pos):
        r"""
        Sets the system's dimensionality.
        """
        if dim is not None:
            self.dim = dim
        elif pos is not None:
            self.dim = len(pos[0])
        elif hop:
            self.dim = len(next(iter(hop.keys())))
        else:
            raise ValueError(
                'No dimension specified and no positions or hoppings are given. The dimensionality of the system cannot be determined.'
            )

        self._zero_vec = tuple([0] * self.dim)

    def _init_hop_pos(self, on_site, hop, pos, contains_cc):
        """
        Sets the hopping terms and positions, mapping the positions to the UC (and changing the hoppings accordingly) if necessary.
        """
        # The double-constructor is needed to avoid a double-constructor in the sparse to-array
        # but still allow for the dtype argument.
        hop = {tuple(key): self._matrix_type(self._matrix_type(value), dtype=complex) for key, value in hop.items()}

        # positions
        if pos is None:
            self.pos = np.zeros((self.size, self.dim))
        elif len(pos) == self.size and all(len(p) == self.dim for p in pos):
            pos, hop = self._map_to_uc(pos, hop)
            self.pos = np.array(pos)  # implicit copy
        else:
            if len(pos) != self.size:
                raise ValueError(
                    "Invalid argument for 'pos': The number of positions must be the same as the size (number of orbitals) of the system."
                )
            raise ValueError(
                "Invalid argument for 'pos': The length of each position must be the same as the dimensionality of the system."
            )

        if contains_cc:
            hop = self._reduce_hop(hop)
        else:
            hop = self._map_hop_positive_R(hop)
        # use partial instead of lambda to allow for pickling
        self.hop = co.defaultdict(self._empty_matrix)
        for R, h_mat in hop.items():
            if not np.any(h_mat):
                continue
            self.hop[R] = self._matrix_type(h_mat)
        # add on-site terms
        if on_site is not None:
            if len(on_site) != self.size:
                raise ValueError(
                    'The number of on-site energies {0} does not match the size of the system {1}'.format(
                        len(on_site), self.size
                    )
                )
            self.hop[self._zero_vec] += 0.5 * self._matrix_type(np.diag(on_site))

    # helpers for _init_hop_pos
    def _map_to_uc(self, pos, hop):
        """
        hoppings in csr format
        """
        uc_offsets = [np.array(np.floor(p), dtype=int) for p in pos]
        # ---- common case: already mapped into the UC ----
        if all([all(o == 0 for o in offset) for offset in uc_offsets]):
            return pos, hop

        # ---- uncommon case: handle mapping ----
        new_pos = [np.array(p) % 1 for p in pos]
        new_hop = co.defaultdict(lambda: np.zeros((self.size, self.size), dtype=complex))
        for R, hop_mat in hop.items():
            hop_mat = np.array(hop_mat)
            for i0, row in enumerate(hop_mat):
                for i1, t in enumerate(row):
                    if t != 0:
                        R_new = tuple(np.array(R, dtype=int) + uc_offsets[i1] - uc_offsets[i0])
                        new_hop[R_new][i0][i1] += t
        new_hop = {key: self._matrix_type(value) for key, value in new_hop.items()}
        return new_pos, new_hop

    @staticmethod
    def _reduce_hop(hop):
        """
        Reduce the full hoppings representation (with cc) to the reduced one (without cc, zero-terms halved).
        """
        # Consistency checks
        for R, mat in hop.items():
            if la.norm(mat - hop.get(tuple(-x for x in R), np.zeros(mat.shape)).T.conjugate()) > 1e-12:
                raise ValueError(
                    'The provided hoppings do not correspond to a hermitian Hamiltonian. hoppings[-R] = hoppings[R].H is not fulfilled.'
                )

        res = dict()
        for R, mat in hop.items():
            try:
                if R[np.nonzero(R)[0][0]] > 0:
                    res[R] = mat
                else:
                    continue
            # zero case
            except IndexError:
                res[R] = 0.5 * mat
        return res

    def _map_hop_positive_R(self, hop):
        """
        Maps hoppings with a negative first non-zero index in R to their positive counterpart.
        """
        new_hop = co.defaultdict(self._empty_matrix)
        for R, mat in hop.items():
            try:
                if R[np.nonzero(R)[0][0]] > 0:
                    new_hop[R] += mat
                else:
                    minus_R = tuple(-x for x in R)
                    new_hop[minus_R] += mat.transpose().conjugate()
            except IndexError:
                # make sure the zero term is also hermitian
                # This only really needed s.t. the representation is unique.
                # The Hamiltonian is anyway made hermitian later.
                new_hop[R] += 0.5 * mat + 0.5 * mat.conjugate().transpose()
        return new_hop

    # end helpers for _init_hop_pos

    def _check_size_hop(self):
        """
        Consistency check for the size of the hopping matrices.
        """
        for h_mat in self.hop.values():
            if not h_mat.shape == (self.size, self.size):
                raise ValueError(
                    'Hopping matrix of shape {0} found, should be ({1},{1}).'.format(h_mat.shape, self.size)
                )

    def _check_dim(self):
        """Consistency check for the dimension of the hoppings and unit cell. The position is checked in _init_hop_pos"""
        for key in self.hop.keys():
            if len(key) != self.dim:
                raise ValueError(
                    'The length of R = {0} does not match the dimensionality of the system ({1})'.format(key, self.dim)
                )
        if self.uc is not None:
            if self.uc.shape != (self.dim, self.dim):
                raise ValueError(
                    'Inconsistend dimension of the unit cell: {0}, does not match the dimensionality of the system ({1})'
                    .format(self.uc.shape, self.dim)
                )

    #---------------- CONSTRUCTORS / (DE)SERIALIZATION ----------------#
    @classmethod
    def from_hop_list(
        cls,
        *,
        hop_list: ty.Iterable[ty.Collection[ty.Union[complex, int, ty.Collection[int]]]] = (),
        size: ty.Optional[int] = None,
        **kwargs
    ) -> "Model":
        """
        Create a :class:`.Model` from a list of hopping terms.

        Parameters
        ----------
        hop_list :
            List of hopping terms. Each hopping term has the form
            [t, orbital_1, orbital_2, R], where

                * ``t``: strength of the hopping
                * ``orbital_1``: index of the first involved orbital
                * ``orbital_2``: index of the second involved orbital
                * ``R``: lattice vector of the unit cell containing the second orbital.
        size :
            Number of states. Defaults to the length of the on-site energies given, if such are given.
        kwargs :
            Any :class:`.Model` keyword arguments.
        """
        if size is None:
            try:
                size = len(kwargs['on_site'])
            except KeyError:
                raise ValueError('No on-site energies and no size given. The size of the system cannot be determined.')

        class _hop:
            """
            POD for hoppings
            """
            def __init__(self):
                self.data = []
                self.row_idx = []
                self.col_idx = []

            def append(self, data, row_idx, col_idx):
                self.data.append(data)
                self.row_idx.append(row_idx)
                self.col_idx.append(col_idx)

        # create data, row_idx, col_idx for setting up the CSR matrices
        hop_list_dict: ty.Mapping[ty.Tuple[int, ...], _hop] = co.defaultdict(_hop)
        R: ty.Collection[int]
        for t, i, j, R in hop_list:  # type: ignore
            R_vec = tuple(R)
            hop_list_dict[R_vec].append(t, i, j)

        # creating CSR matrices
        hop_dict = dict()
        for key, val in hop_list_dict.items():
            hop_dict[key] = sp.csr((val.data, (val.row_idx, val.col_idx)), dtype=complex, shape=(size, size))

        return cls(size=size, hop=hop_dict, **kwargs)

    @staticmethod
    def _read_hr(iterator, ignore_orbital_order=False):
        r"""
        read the number of wannier functions and the hopping entries
        from *hr.dat and converts them into the right format
        """
        next(iterator)  # skip first line
        num_wann = int(next(iterator))
        nrpts = int(next(iterator))

        # get degeneracy points
        deg_pts = []
        # order in zip important because else the next data element is consumed
        for _, line in zip(range(int(np.ceil(nrpts / 15))), iterator):
            deg_pts.extend(int(x) for x in line.split())
        assert len(deg_pts) == nrpts

        num_wann_square = num_wann**2

        def to_entry(line, i):
            """Turns a line (string) into a hop_list entry"""
            entry = line.split()
            orbital_a = int(entry[3]) - 1
            orbital_b = int(entry[4]) - 1
            # test consistency of orbital numbers
            if not ignore_orbital_order:
                if not (orbital_a == i % num_wann) and (orbital_b == (i % num_wann_square) // num_wann):
                    raise ValueError("Inconsistent orbital numbers in line '{}'".format(line))
            return [(float(entry[5]) + 1j * float(entry[6])) / (deg_pts[i // num_wann_square]), orbital_a, orbital_b,
                    [int(x) for x in entry[:3]]]

        # skip random empty lines
        lines_nonempty = (l for l in iterator if l.strip())
        hop_list = (to_entry(line, i) for i, line in enumerate(lines_nonempty))

        return num_wann, hop_list

    def to_hr_file(self, hr_file: str) -> None:
        """
        Writes to a file, using Wannier90's ``*_hr.dat`` format.

        Parameters
        ----------
        hr_file :
            Path of the output file


        .. note :: The ``*_hr.dat`` format does not contain information
            about the position of the atoms or the shape of the unit
            cell. Consequently, this information is lost when saving the
            model in this format.

        .. warning :: The ``*_hr.dat`` format does not preserve the full
            precision of the hopping strengths. This could lead to
            numerical errors.
        """
        with open(hr_file, 'w') as f:
            f.write(self.to_hr())

    def to_hr(self) -> str:
        """
        Returns a string containing the model in Wannier90's
        ``*_hr.dat`` format.

        .. note :: The ``*_hr.dat`` format does not contain information about the position of the atoms or the shape of the unit cell. Consequently, this information is lost when saving the model in this format.

        .. warning :: The ``*_hr.dat`` format does not preserve the full precision of the hopping strengths. This could lead to numerical errors.
        """
        lines = []
        tagline = ' created by the TBmodels package    ' + time.strftime('%a, %d %b %Y %H:%M:%S %Z')
        lines.append(tagline)
        lines.append('{0:>12}'.format(self.size))
        num_g = len(self.hop.keys()) * 2 - 1
        if num_g <= 0:
            raise ValueError('Cannot print empty model to hr format.')
        lines.append('{0:>12}'.format(num_g))
        tmp = ''
        for i in range(num_g):
            if tmp != '' and i % 15 == 0:
                lines.append(tmp)
                tmp = ''
            tmp += '    1'
        lines.append(tmp)

        # negative
        for R in reversed(sorted(self.hop.keys())):
            if R != self._zero_vec:
                minus_R = tuple(-x for x in R)
                lines.extend(self._mat_to_hr(minus_R, self.hop[R].conjugate().transpose()))
        # zero
        if self._zero_vec in self.hop.keys():
            lines.extend(
                self._mat_to_hr(
                    self._zero_vec, self.hop[self._zero_vec] + self.hop[self._zero_vec].conjugate().transpose()
                )
            )
        # positive
        for R in sorted(self.hop.keys()):
            if R != self._zero_vec:
                lines.extend(self._mat_to_hr(R, self.hop[R]))

        return '\n'.join(lines)

    @staticmethod
    def _mat_to_hr(R, mat):
        """
        Creates the ``*_hr.dat`` string for a single hopping matrix.
        """
        lines = []
        mat = np.array(mat).T  # to be consistent with W90's ordering
        for j, column in enumerate(mat):
            for i, t in enumerate(column):
                lines.append(
                    '{0[0]:>5}{0[1]:>5}{0[2]:>5}{1:>5}{2:>5}{3.real:>22.14f}{3.imag:>22.14f}'.format(
                        R, i + 1, j + 1, t
                    )
                )
        return lines

    @classmethod
    def from_wannier_folder(cls, folder: str = '.', prefix: str = 'wannier', **kwargs) -> "Model":
        """
        Create a :class:`.Model` instance from Wannier90 output files,
        given the folder containing the files and file prefix.

        Parameters
        ----------
        folder :
            Directory containing the Wannier90 output files.
        prefix :
            Prefix of the Wannier90 output files.
        kwargs :
            Keyword arguments passed to :meth:`.from_wannier_files`. If
            input files are explicitly given, they take precedence over
            those found in the ``folder``.
        """
        common_path = os.path.join(folder, prefix)
        input_files = dict()
        input_files['hr_file'] = common_path + '_hr.dat'

        for key, suffix in [
            ('win_file', '.win'),
            ('wsvec_file', '_wsvec.dat'),
            ('xyz_file', '_centres.xyz'),
        ]:
            filename = common_path + suffix
            if os.path.isfile(filename):
                input_files[key] = filename

        return cls.from_wannier_files(**co.ChainMap(kwargs, input_files))

    @classmethod
    def from_wannier_files(  # pylint: disable=too-many-locals
        cls,
        *,
        hr_file: str,
        wsvec_file: ty.Optional[str] = None,
        xyz_file: ty.Optional[str] = None,
        win_file: ty.Optional[str] = None,
        h_cutoff: float = 0.,
        ignore_orbital_order: bool = False,
        pos_kind: str = 'wannier',
        **kwargs
    ) -> "Model":
        """
        Create a :class:`.Model` instance from Wannier90 output files.

        Parameters
        ----------
        hr_file :
            Path of the ``*_hr.dat`` file. Together with the
            ``*_wsvec.dat`` file, this determines the hopping terms.
        wsvec_file :
            Path of the ``*_wsvec.dat`` file. This file determines the
            remapping of hopping terms when ``use_ws_distance`` is used
            in the Wannier90 calculation.
        xyz_file :
            Path of the ``*_centres.xyz`` file. This file is used to
            determine the positions of the orbitals, from the Wannier
            centers given by Wannier90.
        win_file :
            Path of the ``*.win`` file. This file is used to determine
            the unit cell.
        h_cutoff :
            Cutoff value for the hopping strength. Hoppings with a
            smaller absolute value are ignored.
        ignore_orbital_order :
            Do not throw an error when the order of orbitals does not
            match what is expected from the Wannier90 output.
        pos_kind :
            Determines how positions are assinged to orbitals. Valid
            options are `wannier` (use Wannier centres) or
            `nearest_atom` (map to nearest atomic position).
        kwargs :
            :class:`.Model` keyword arguments.
        """

        if win_file is not None:
            if 'uc' in kwargs:
                raise ValueError(
                    "Ambiguous unit cell: It can be given either via 'uc' or the 'win_file' keywords, but not both."
                )
            with open(win_file, 'r') as f:
                kwargs['uc'] = cls._read_win(f)['unit_cell_cart']

        if xyz_file is not None:
            if 'pos' in kwargs:
                raise ValueError(
                    "Ambiguous orbital positions: The positions can be given either via the 'pos' or the 'xyz_file' keywords, but not both."
                )
            if 'uc' not in kwargs:
                raise ValueError(
                    "Positions cannot be read from .xyz file without unit cell given: Transformation from cartesian to reduced coordinates not possible. Specify the unit cell using one of the keywords 'uc' or 'win_file'."
                )
            with open(xyz_file, 'r') as f:
                wannier_pos_list_cartesian, atom_list_cartesian = cls._read_xyz(f)
                wannier_pos_cartesian = np.array(wannier_pos_list_cartesian)
                atom_pos_cartesian = np.array([a.pos for a in atom_list_cartesian])
                if pos_kind == 'wannier':
                    pos_cartesian = wannier_pos_cartesian
                elif pos_kind == 'nearest_atom':
                    pos_cartesian = []
                    for p in wannier_pos_cartesian:
                        p_reduced = la.solve(kwargs['uc'].T, np.array(p).T).T
                        T_base = np.floor(p_reduced)
                        all_atom_pos = np.array([
                            kwargs['uc'].T @ (T_base + T_shift) + atom_pos for atom_pos in atom_pos_cartesian
                            for T_shift in itertools.product([-1, 0, 1], repeat=3)
                        ])
                        distances = la.norm(p - all_atom_pos, axis=-1)
                        pos_cartesian.append(all_atom_pos[np.argmin(distances)])
                else:
                    raise ValueError(
                        "Invalid value '{}' for 'pos_kind', must be 'wannier' or 'nearest_atom'".format(pos_kind)
                    )
                kwargs['pos'] = la.solve(kwargs['uc'].T, np.array(pos_cartesian).T).T

        with open(hr_file, 'r') as f:
            num_wann, hop_entries = cls._read_hr(f, ignore_orbital_order=ignore_orbital_order)
            hop_entries = (hop for hop in hop_entries if abs(hop[0]) > h_cutoff)

            if wsvec_file is not None:
                with open(wsvec_file, 'r') as f:
                    # wsvec_mapping is not a generator because it doesn't have
                    # the same order as the hoppings in _hr.dat
                    # This could still be done, but would be more complicated.
                    wsvec_generator = cls._async_parse(cls._read_wsvec(f), chunksize=num_wann)

                    def remap_hoppings(hop_entries):
                        for t, orbital_1, orbital_2, R in hop_entries:
                            try:
                                next(wsvec_generator)
                            # Explicitly catching this because 'remap_hoppings'
                            # is also a generator.
                            except StopIteration as exc:
                                raise ValueError("The 'wsvec_generator' stopped prematurely.") from exc
                            T_list = wsvec_generator.send((orbital_1, orbital_2, tuple(R)))
                            N = len(T_list)
                            for T in T_list:
                                # not using numpy here increases performance
                                yield (t / N, orbital_1, orbital_2, tuple(r + t for r, t in zip(R, T)))

                    hop_entries = remap_hoppings(hop_entries)
                    return cls.from_hop_list(size=num_wann, hop_list=hop_entries, **kwargs)

            return cls.from_hop_list(size=num_wann, hop_list=hop_entries, **kwargs)

    @staticmethod
    def _async_parse(iterator, chunksize=1):
        """
        Helper function to get values from a (key, value) iterator
        out of order without having to exhaust the iterator from the start.
        The desired key needs to be sent to this generator, and it
        will go through the `iterator` until that key is found. Pairs
        for which the key has not yet been requested are stored in a
        temporary dictionary.
        """
        mapping = dict()
        stopped = False
        while True:
            # get the desired key
            key = yield
            while True:
                try:
                    # key found
                    yield mapping.pop(key)
                    break
                except KeyError as e:
                    if stopped:
                        # avoid infinte loop in true KeyError
                        raise e
                    for _ in range(chunksize):
                        try:
                            # parse new data
                            newkey, newval = next(iterator)
                            mapping[newkey] = newval
                        except StopIteration:
                            stopped = True
                            break

    @staticmethod
    def _read_wsvec(iterator):
        """
        Generator that parses the content of the *_wsvec.dat file.
        """
        # skip comment line
        try:
            next(iterator)
        except StopIteration as exc:
            raise ValueError("The 'wsvec' iterator is empty.") from exc
        for first_line in iterator:
            *R, o1, o2 = (int(x) for x in first_line.split())
            # in our convention, orbital indices start at 0.
            key = (o1 - 1, o2 - 1, tuple(R))
            try:
                N = int(next(iterator))
                val = [tuple(int(x) for x in next(iterator).split()) for _ in range(N)]
            except StopIteration as exc:
                raise ValueError('Incomplete wsvec iterator.') from exc
            yield key, val

    @staticmethod
    def _read_xyz(iterator):
        """Reads the content of a .xyz file"""
        # This functionality exists within pymatgen, so it might make sense
        # to use that if we anyway want pymatgen as a dependency.
        N = int(next(iterator))
        next(iterator)  # skip comment line
        wannier_centres = []
        atom_positions = []
        AtomPosition = co.namedtuple('AtomPosition', ['kind', 'pos'])
        for l in iterator:
            kind, *pos = l.split()
            pos = tuple(float(x) for x in pos)
            if kind == 'X':
                wannier_centres.append(pos)
            else:
                atom_positions.append(AtomPosition(kind=kind, pos=pos))
        assert len(wannier_centres) + len(atom_positions) == N
        return wannier_centres, atom_positions

    @staticmethod
    def _read_win(iterator):
        """
        Takes an iterator representing the Wannier90 .win file lines,
        and returns a mapping of its content.
        """
        lines = (l.split('!')[0] for l in iterator)
        lines = (l.strip() for l in lines)
        lines = (l for l in lines if l)
        lines = (l.lower() for l in lines)

        split_token = re.compile('[\t :=]+')

        mapping = {}
        for line in lines:
            if line.startswith('begin'):
                key = split_token.split(line[5:].strip(' :='), 1)[0]
                val = []
                while True:
                    line = next(lines)
                    if line.startswith('end'):
                        end_key = split_token.split(line[3:].strip(' :='), 1)[0]
                        assert end_key == key
                        break
                    val.append(line)
                mapping[key] = val
            else:
                key, val = split_token.split(line, 1)
                mapping[key] = val

        # here we can continue parsing the individual keys as needed
        if 'length_unit' in mapping:
            length_unit = mapping['length_unit'].strip().lower()
        else:
            length_unit = 'ang'
        mapping['length_unit'] = length_unit

        if 'unit_cell_cart' in mapping:
            uc_input = mapping['unit_cell_cart']
            # handle the case when the unit is explicitly given
            if len(uc_input) == 4:
                unit, *uc_input = uc_input
                # unit = unit[0]
            else:
                unit = length_unit
            val = [[float(x) for x in split_token.split(line)] for line in uc_input]
            val = np.array(val).reshape(3, 3)
            if unit == 'bohr':
                val *= 0.52917721092
            mapping['unit_cell_cart'] = val

        return mapping

    def to_kwant_lattice(self):
        """
        Returns a kwant lattice corresponding to the current model. Orbitals with the same position are grouped into the same Monoatomic sublattice.

        .. note :: The TBmodels - Kwant interface is experimental. Use it with caution.
        """
        import kwant  # pylint: disable=import-outside-toplevel
        sublattices = self._get_sublattices()
        uc = self.uc if self.uc is not None else np.eye(self.dim)
        # get sublattice positions in cartesian coordinates
        pos_abs = np.dot(np.array([sl.pos for sl in sublattices]), uc)
        return kwant.lattice.general(prim_vecs=uc, basis=pos_abs)

    def add_hoppings_kwant(self, kwant_sys):
        """
        Sets the on-site energies and hopping terms for an existing kwant system to those of the :class:`.Model`.

        .. note :: The TBmodels - Kwant interface is experimental. Use it with caution.
        """
        import kwant  # pylint: disable=import-outside-toplevel
        sublattices = self._get_sublattices()
        kwant_sublattices = self.to_kwant_lattice().sublattices

        # handle R = 0 case (on-site)
        on_site_mat = copy.deepcopy(self._array_cast(self.hop[self._zero_vec]))
        on_site_mat += on_site_mat.conjugate().transpose()
        # R = 0 terms within a sublattice (on-site)
        for site in kwant_sys.sites():
            for i, latt in enumerate(kwant_sublattices):
                if site.family == latt:
                    indices = sublattices[i].indices
                    kwant_sys[site] = on_site_mat[np.ix_(indices, indices)]
                    break
            # site doesn't belong to any sublattice
            else:
                # TODO: check if there is a legitimate use case which triggers this
                raise ValueError('Site {} did not match any sublattice.'.format(site))

        # R = 0 terms between different sublattices
        for i, s1 in enumerate(sublattices):
            for j, s2 in enumerate(sublattices):
                if i == j:
                    # handled above
                    continue
                kwant_sys[kwant.builder.HoppingKind(self._zero_vec, kwant_sublattices[i],
                                                    kwant_sublattices[j])] = on_site_mat[np.ix_(s1.indices, s2.indices)]

        # R != 0 terms
        for R, mat in self.hop.items():
            mat = self._array_cast(mat)
            # special case R = 0 handled already
            if R == self._zero_vec:
                continue
            minus_R = tuple(-np.array(R))
            for i, s1 in enumerate(sublattices):
                for j, s2 in enumerate(sublattices):
                    sub_matrix = mat[np.ix_(s1.indices, s2.indices)]
                    # TODO: check "signs"
                    kwant_sys[kwant.builder.HoppingKind(minus_R, kwant_sublattices[i],
                                                        kwant_sublattices[j])] = sub_matrix
                    kwant_sys[kwant.builder.HoppingKind(R, kwant_sublattices[j],
                                                        kwant_sublattices[i])] = np.transpose(np.conj(sub_matrix))
        return kwant_sys

    def _get_sublattices(self):
        """
        Helper function to group indices of orbitals which have the same
        position into sublattices.
        """
        Sublattice = co.namedtuple('Sublattice', ['pos', 'indices'])
        sublattices = []
        for i, p_orb in enumerate(self.pos):
            # try to match an existing sublattice
            for sub_pos, sub_indices in sublattices:
                if np.isclose(p_orb, sub_pos, rtol=0).all():
                    sub_indices.append(i)
                    break
            # create new sublattice
            else:
                sublattices.append(Sublattice(pos=p_orb, indices=[i]))
        return sublattices

    def construct_kdotp(self, k: ty.Collection[float], order: int):
        """
        Construct a k.p model around a given k-point. This is done by explicitly
        evaluating the derivatives which make up the Taylor expansion of the k.p
        models.

        This method can currently only construct models using
        `convention 2  <http://www.physics.rutgers.edu/pythtb/_downloads/pythtb-formalism.pdf>`_
        for the Hamiltonian.

        Parameters
        ----------
        k :
            The k-point around which the k.p model is constructed.
        order :
            The order (sum of powers) to which the Taylor expansion is
            performed.
        """
        # foo : ty.Collection[int] = (1, 2, 3)
        # taylor_coefficients : ty.Dict[ty.Tuple[int, ...], ty.Any] = dict()
        taylor_coefficients = dict()
        if order < 0:
            raise ValueError('The order for the k.p model must be positive.')
        k_powers: ty.Tuple[int, ...]
        for k_powers in itertools.product(range(order + 1), repeat=self.dim):
            curr_order = sum(k_powers)
            if curr_order > order:
                continue
            taylor_coefficients[k_powers] = (
                (2j * np.pi)**curr_order / np.prod(factorial(k_powers, exact=True))
            ) * sum((
                np.prod(np.array(R)**np.array(k_powers)) * np.exp(2j * np.pi * np.dot(k, R)) * self._array_cast(mat) +
                np.prod((-np.array(R))**np.array(k_powers)) * np.exp(-2j * np.pi * np.dot(k, R)) *
                self._array_cast(mat).T.conj() for R, mat in self.hop.items()
            ), np.zeros((self.size, self.size), dtype=complex))
        return KdotpModel(taylor_coefficients=taylor_coefficients)

    @classmethod
    def from_hdf5_file(cls, hdf5_file: str, **kwargs) -> "Model":  # pylint: disable=arguments-differ
        """
        Returns a :class:`.Model` instance read from a file in HDF5
        format.

        Parameters
        ----------
        hdf5_file :
            Path of the input file.
        kwargs :
            :class:`.Model` keyword arguments. Explicitly specified
            keywords take precedence over those given in the HDF5 file.
        """
        with h5py.File(hdf5_file, 'r') as f:
            return cls.from_hdf5(f, **kwargs)

    @classmethod
    def from_hdf5(cls, hdf5_handle, **kwargs) -> "Model":  # pylint: disable=arguments-differ
        # For compatibility with a development version which created a top-level
        # 'tb_model' attribute.
        try:
            tb_model_group = hdf5_handle['tb_model']
        except KeyError:
            tb_model_group = hdf5_handle
        new_kwargs: ty.Dict[str, ty.Any] = {}
        new_kwargs['hop'] = {}

        for key in ['uc', 'occ', 'size', 'dim', 'pos', 'sparse']:
            if key in tb_model_group:
                new_kwargs[key] = tb_model_group[key][()]

        if 'hop' not in kwargs:
            for group in tb_model_group['hop'].values():
                R = tuple(group['R'])
                if new_kwargs['sparse']:
                    new_kwargs['hop'][R] = sp.csr((group['data'], group['indices'], group['indptr']),
                                                  shape=group['shape'])
                else:
                    new_kwargs['hop'][R] = np.array(group['mat'])
            new_kwargs['contains_cc'] = False
        return cls(**co.ChainMap(kwargs, new_kwargs))

    def to_hdf5(self, hdf5_handle):
        if self.uc is not None:
            hdf5_handle['uc'] = self.uc
        if self.occ is not None:
            hdf5_handle['occ'] = self.occ
        hdf5_handle['size'] = self.size
        hdf5_handle['dim'] = self.dim
        hdf5_handle['pos'] = self.pos
        hdf5_handle['sparse'] = self._sparse
        hop = hdf5_handle.create_group('hop')
        for i, (R, mat) in enumerate(self.hop.items()):
            group = hop.create_group(str(i))
            group['R'] = R
            if self._sparse:
                group['data'] = mat.data
                group['indices'] = mat.indices
                group['indptr'] = mat.indptr
                group['shape'] = mat.shape
            else:
                group['mat'] = mat

    def __repr__(self):
        return ' '.join(
            'tbmodels.Model(hop={1}, pos={0.pos!r}, uc={0.uc!r}, occ={0.occ}, contains_cc=False)'.format(
                self, dict(self.hop)
            ).replace('\n', ' ').replace('array', 'np.array').split()
        )

    #---------------- BASIC FUNCTIONALITY ----------------------------------#
    @property
    def reciprocal_lattice(self):
        """An array containing the reciprocal lattice vectors as rows."""
        return None if self.uc is None else 2 * np.pi * la.inv(self.uc).T

    def hamilton(self, k: ty.Collection[float], convention: int = 2) -> np.ndarray:
        """
        Calculates the Hamilton matrix for a given k-point.

        Parameters
        ----------
        k :
            The k-point at which the Hamiltonian is evaluated.
        convention :
            Choice of convention to calculate the Hamilton matrix. See
            explanation in `the PythTB documentation
            <http://www.physics.rutgers.edu/pythtb/_downloads/pythtb-formalism.pdf>`_ .
            Valid choices are 1 or 2.
        """
        if convention not in [1, 2]:
            raise ValueError("Invalid value '{}' for 'convention': must be either '1' or '2'".format(convention))
        k = np.array(k, ndmin=1)
        H = sum((self._array_cast(hop) * np.exp(2j * np.pi * np.dot(R, k)) for R, hop in self.hop.items()),
                np.zeros((self.size, self.size), dtype=complex))
        H += H.conjugate().T
        if convention == 1:
            pos_exponential = np.array([[np.exp(2j * np.pi * np.dot(p, k)) for p in self.pos]])
            H = pos_exponential.conjugate().transpose() * H * pos_exponential
        return H

    def eigenval(self, k: ty.Collection[float]) -> np.ndarray:
        """
        Returns the eigenvalues at a given k point.

        Parameters
        ----------
        k :
            The k-point at which the Hamiltonian is evaluated.
        """
        return la.eigvalsh(self.hamilton(k))

    def dos(
        self,
        kmesh: ty.Collection[int],
        energy_range=None,
        smr_index=0,
        smr_width=0.1
    ) -> np.ndarray:
        """Calculate dos in the specified energy range

        default smearing is gaussian

        :param energy_range: [energy_min, energy_max, energy_step]
        :type energy_range: ty.Collection[float]
        :return: dos at specified energy points
        :rtype: np.ndarray
        """
        if len(kmesh) != 3:
            raise ValueError("Invalid kpoint mesh:" + kmesh)

        from itertools import product
        num_kpts = np.product(kmesh)
        kweight = 1.0 / num_kpts
        kpt_x = np.linspace(0, 1, kmesh[0], endpoint=False)
        kpt_y = np.linspace(0, 1, kmesh[1], endpoint=False)
        kpt_z = np.linspace(0, 1, kmesh[2], endpoint=False)
        kpt_list = product(kpt_x, kpt_y, kpt_z)

        # currently only works for no-spin case
        num_elec_per_state = 2
        # smearing_cutoff = 10.0
        min_smearing_binwidth_ratio = 2.0

        eigs = np.zeros((num_kpts, self.size))
        for ik, kpt in enumerate(kpt_list):
            eigs[ik] = self.eigenval(kpt)

        if energy_range is None:
            emax = eigs.max() - 0.5
            emin = eigs.min() + 0.5
            energy_step = min((emax-emin)/200, 0.05)
            energy_list = np.arange(emin, emax, energy_step)
        else:
            if len(energy_range) != 3:
                raise ValueError("Invalid energy range {}".format(str(energy_range)))
            energy_step = energy_range[2]
            energy_list = np.arange(*energy_range)

        if (smr_width / energy_step) < min_smearing_binwidth_ratio:
            do_smearing = False
        else:
            do_smearing = True

        def w0gauss(x: ty.Collection[float]) -> ty.Collection[float]:
            """from wannier90/src/utility.F90, fully accept numpy.array

            the derivative of utility_wgauss:  an approximation to the delta function
            
            (n>=0) : derivative of the corresponding Methfessel-Paxton utility_wgauss
            
            (n=-1 ): derivative of cold smearing:
                        1/sqrt(pi)*exp(-(x-1/sqrt(2))**2)*(2-sqrt(2)*x)
            
            (n=-99): derivative of Fermi-Dirac function: 0.5/(1.0+cosh(x))
            
            :param x: [description]
            :type x: np.array
            :param smr_index: the order of the smearing function
            :type smr_index: int
            :return: [description]
            :rtype: np.array
            """

            w0gauss = np.zeros(x.shape)

            # Fermi-Dirac smearing
            sqrtpm1 = 1.0 / np.sqrt(np.pi)

            if smr_index == -99:
                # in order to avoid problems for large values of x in the e
                arg = np.abs(x) < 36.0
                w0gauss[arg] = 1.00 / (2.00 + np.exp(-x[arg]) + np.exp(+x[arg]))
                return w0gauss

            # cold smearing  (Marzari-Vanderbilt)
            if smr_index == -1:
                arg = (x - 1 / np.sqrt(2))**2
                arg[arg > 200] = 200
                w0gauss = sqrtpm1 * np.exp(-arg) * (2 - np.sqrt(2) * x)
                return w0gauss

            if (smr_index > 10) or (smr_index < 0):
                raise ValueError('higher order smearing is untested and unstable')

            # Methfessel-Paxton
            arg = x**2
            arg[arg > 200] = 200
            w0gauss = np.exp(-arg) * sqrtpm1

            if smr_index == 0:
                return w0gauss

            hd = 0.0
            hp = np.exp(-arg)
            ni = 0.0
            a = sqrtpm1
            for i in range(1, smr_index + 1):
                hd = 2 * x * hp - 2 * ni * hd
                ni = ni + 1
                a = -a / (i * 4)
                hp = 2 * x * hd - 2 * ni * hp
                ni = ni + 1
                w0gauss = w0gauss + a * hp

            return w0gauss

        def get_dos_k(ik: int) -> np.ndarray:
            """calculates the contribution to the DOS of a single k point
            
            :param energy_list: [description]
            :type energy_list: [type]
            :param k: [description]
            :type k: [type]
            :param smr_index: [description]
            :type smr_index: int
            :param smr_width: [description]
            :type smr_width: [type]
            """
            eig_k = eigs[ik].reshape((1, -1)) # in row
            e_list = energy_list.reshape((-1, 1)) # in column

            dos_k = np.zeros(shape=len(energy_list))
            
            mat_e_minus_eig = e_list - eig_k # use np broadcasting
            if (do_smearing):
                arg = mat_e_minus_eig / smr_width
                dos_k = np.sum(w0gauss(arg) / smr_width, axis=1)
            else:
                # find grid in energy_list which is nearest to each eigenvalue
                arg = np.argmin(np.abs(mat_e_minus_eig), axis=0)
                ind, fac = np.unique(arg, return_counts=True)
                dos_k[ind] = 1.0 / energy_step * fac

            return dos_k * num_elec_per_state

            # f = open("kpt"+str(i), mode='w')#
            # f.write("{} {} {}\n".format(k[0], k[1], k[2]))#
            # for d in dos_k:
            #     f.write("{}\n".format(d))#
            # f.close()

            # f = open("eig"+str(i), mode='w')#
            # f.write("{} {} {}\n".format(k[0], k[1], k[2]))#
            # for d in eig_k:
            #     f.write("{}\n".format(d))#
            # f.close()

        dos_all = np.zeros(shape=len(energy_list))
        for ik in range(num_kpts):
            dos_k = get_dos_k(ik)
            dos_all = dos_all + dos_k * kweight
        return (energy_list, dos_all)

    #-------------------MODIFYING THE MODEL ----------------------------#
    def add_hop(self, overlap: complex, orbital_1: int, orbital_2: int, R: ty.Collection[int]):
        r"""
        Adds a hopping term with a given overlap (hopping strength) from
        ``orbital_2`` (:math:`o_2`), which lies in the unit cell pointed
        to by ``R``, to ``orbital_1`` (:math:`o_1`) which is in the home
        unit cell. In other words, ``overlap`` is the matrix element
        :math:`\mathcal{H}_{o_1,o_2}(\mathbf{R}) = \langle o_1, \mathbf{0} | \mathcal{H} | o_2, \mathbf{R} \rangle`.

        The complex conjugate of the hopping is added automatically.
        That is, the matrix element
        :math:`\langle o_2, \mathbf{R} | \mathcal{H} | o_1, \mathbf{0} \rangle`
        does not have to be added manually.

        .. note::
            This means that adding a hopping of overlap :math:`\epsilon`
            between an orbital and itself in the home unit cell
            increases the orbitals on-site energy by :math:`2 \epsilon`.


        Parameters
        ----------
        overlap :
            Strength of the hopping term (in energy units).
        orbital_1 :
            Index of the first orbital.
        orbital_2 :
            Index of the second orbital.
        R :
            Lattice vector pointing to the unit cell where ``orbital_2``
            lies.

        .. warning::
            The positions given in the constructor of :class:`.Model`
            are automatically mapped into the home unit cell. This has
            to be taken into account when determining ``R``.

        """
        R = tuple(R)
        if len(R) != self.dim:
            raise ValueError('Dimension of R ({}) does not match the model dimension ({})'.format(len(R), self.dim))

        mat = np.zeros((self.size, self.size), dtype=complex)
        nonzero_idx = np.nonzero(R)[0]
        if nonzero_idx.size == 0:
            mat[orbital_1, orbital_2] += overlap / 2.
            mat[orbital_2, orbital_1] += overlap.conjugate() / 2.
        elif R[nonzero_idx[0]] > 0:
            mat[orbital_1, orbital_2] += overlap
        else:
            R = tuple(-x for x in R)
            mat[orbital_2, orbital_1] += overlap.conjugate()
        self.hop[R] += self._matrix_type(mat)

    def add_on_site(self, on_site: ty.Collection[float]):
        """
        Adds on-site energy to the orbitals. This adds to the existing
        on-site energy, and does not erase it.

        Parameters
        ----------
        on_site :
            On-site energies. This must be a sequence of real numbers, of the same length as the number of orbitals
        """
        if self.size != len(on_site):
            raise ValueError(
                'The number of on-site energy terms should be {}, but is {}.'.format(self.size, len(on_site))
            )
        for orbital, energy in enumerate(on_site):
            self.add_hop(energy / 2., orbital, orbital, self._zero_vec)

    def _empty_matrix(self):
        """Returns an empty matrix, either sparse or dense according to the current setting. The size is determined by the system's size"""
        return self._matrix_type(np.zeros((self.size, self.size), dtype=complex))

    def set_sparse(self, sparse: bool = True):
        """
        Defines whether sparse or dense matrices should be used to
        represent the system, and changes the system accordingly if
        needed.

        Parameters
        ----------
        sparse :
            Flag to determine whether the system is set to be sparse
            (``True``) or dense (``False``).
        """
        # check if the right sparsity is alredy set
        # when using from __init__, self._sparse is not set
        with contextlib.suppress(AttributeError):
            if sparse == self._sparse:
                return

        self._sparse = sparse
        if sparse:
            self._matrix_type = sp.csr
        else:
            self._matrix_type = np.array

        # change existing matrices
        with contextlib.suppress(AttributeError):
            for k, v in self.hop.items():
                self.hop[k] = self._matrix_type(v)

    # If Python 3.4 support is dropped this could be made more straightforwardly
    # However, for now the default pickle protocol (and thus multiprocessing)
    # does not support that.
    def _array_cast(self, x):
        """Casts a matrix type to a numpy array."""
        if self._sparse:
            return np.array(x)
        else:
            return x

    #-------------------CREATING DERIVED MODELS-------------------------#
    #---- arithmetic operations ----#
    @property
    def _input_kwargs(self):
        return dict(hop=self.hop, pos=self.pos, occ=self.occ, uc=self.uc, contains_cc=False, sparse=self._sparse)

    def symmetrize(self, symmetries: ty.Sequence[sr.SymmetryOperation], full_group: bool = False) -> "Model":
        """
        Returns a model which is symmetrized w.r.t. the given
        symmetries. This is done by performing a group average over the
        symmetry group.

        Parameters
        ----------
        symmetries :
            Symmetries which the symmetrized model should respect.
        full_group :
            Specifies whether the given symmetries represent the full
            symmetry group, or only a subset from which the full
            symmetry group is generated.
        """
        if full_group:
            new_model = self._apply_operation(symmetries[0])
            return 1 / len(symmetries) * sum((self._apply_operation(s) for s in symmetries[1:]), new_model)
        else:
            new_model = self
            for sym in symmetries:
                order = sym.get_order()
                sym_pow = sym
                tmp_model = new_model
                for _ in range(1, order):
                    tmp_model += new_model._apply_operation(sym_pow)  # pylint: disable=protected-access
                    sym_pow @= sym
                new_model = 1 / order * tmp_model
            return new_model

    def _apply_operation(self, symmetry_operation) -> "Model":  # pylint: disable=too-many-locals
        """
        Helper function to apply a symmetry operation to the model.
        """
        # apply symmetry operation on sublattice positions
        sublattices = self._get_sublattices()

        new_sublattice_pos = [symmetry_operation.real_space_operator.apply(latt.pos) for latt in sublattices]

        # match to a known sublattice position to determine the shift vector
        uc_shift = []
        for new_pos in new_sublattice_pos:
            nearest_R = np.array(np.rint(new_pos), dtype=int)
            # the new position must be in a neighbouring UC
            valid_shifts = []
            for T in itertools.product(range(-1, 2), repeat=self.dim):
                shift = nearest_R + T
                if any(np.isclose(new_pos - shift, latt.pos).all() for latt in sublattices):
                    valid_shifts.append(tuple(shift))
            if not valid_shifts:
                raise ValueError('New position {} does not match any known sublattice'.format(new_pos))
            if len(valid_shifts) > 1:
                raise ValueError(
                    'Ambiguity error: New position {} matches more than one known sublattice'.format(new_pos)
                )
            uc_shift.append(valid_shifts[0])

        # setting up the indices to slice the hopping matrices
        hop_shifts_idx: ty.Dict[ty.Tuple[int, ...], ty.Tuple[ty.List[int],
                                                             ty.List[int]]] = co.defaultdict(lambda: ([], []))
        for (i, Ti), (j, Tj) in itertools.product(enumerate(uc_shift), repeat=2):
            shift = tuple(np.array(Tj) - np.array(Ti))
            for idx1, idx2 in itertools.product(sublattices[i].indices, sublattices[j].indices):
                hop_shifts_idx[shift][0].append(idx1)
                hop_shifts_idx[shift][1].append(idx2)

        # create hoppings with shifted R (by uc_shift[j] - uc_shift[i])
        new_hop: ty.Dict[ty.Tuple[int, ...], ty.Any] = co.defaultdict(self._empty_matrix)
        for R, mat in self.hop.items():
            R_transformed = np.array(np.rint(np.dot(symmetry_operation.rotation_matrix, R)), dtype=int)
            for shift, (idx1, idx2) in hop_shifts_idx.items():
                new_R = tuple(np.array(R_transformed) + np.array(shift))
                new_hop[new_R][idx1, idx2] += mat[idx1, idx2]

        # apply D(g) ... D(g)^-1 (since D(g) is unitary: D(g)^-1 == D(g)^H)
        for R in new_hop.keys():
            sym_op = np.array(symmetry_operation.repr.matrix).astype(complex)
            if symmetry_operation.repr.has_cc:
                new_hop[R] = np.conj(new_hop[R])
            new_hop[R] = np.dot(sym_op, np.dot(new_hop[R], np.conj(np.transpose(sym_op))))

        return Model(**co.ChainMap(dict(hop=new_hop), self._input_kwargs))

    def slice_orbitals(self, slice_idx: ty.List[int]) -> "Model":
        """
        Returns a new model with only the orbitals as given in the
        ``slice_idx``. This can also be used to re-order the orbitals.

        Parameters
        ----------
        slice_idx :
            Orbital indices that will be in the resulting model.
        """
        new_pos = self.pos[tuple(slice_idx), :]
        new_hop = {key: np.array(val)[np.ix_(slice_idx, slice_idx)] for key, val in self.hop.items()}
        return Model(**co.ChainMap(dict(hop=new_hop, pos=new_pos), self._input_kwargs))

    @classmethod
    def join_models(cls, *models: "Model") -> "Model":
        """
        Creates a tight-binding model which contains all orbitals of the
        given input models. The orbitals are ordered by model, such that
        the resulting Hamiltonian is block-diagonal.

        Parameters
        ----------
        models :
            Models which should be joined together.
        """
        if not models:
            raise ValueError('At least one model must be given.')

        first_model = models[0]
        # check dim
        if not _check_compatibility.check_dim(*models):
            raise ValueError('Model dimensions do not match.')
        new_dim = first_model.dim

        # check uc compatibility
        if not _check_compatibility.check_uc(*models):
            raise ValueError('Model unit cells do not match.')
        new_uc = first_model.uc

        # join positions (must either all be set, or all None)
        pos_list = list(m.pos for m in models)
        if any(pos is None for pos in pos_list):
            if not all(pos is None for pos in pos_list):
                raise ValueError('Either all or no positions must be set.')
            new_pos = None
        else:
            new_pos = np.concatenate(pos_list)

        # add occ (is set to None if any model has occ=None)
        occ_list = list(m.occ for m in models)
        if any(occ is None for occ in occ_list):
            new_occ = None
        else:
            new_occ = sum(occ_list)

        # combine hop
        all_R: ty.Set[ty.Tuple[int, ...]] = set()
        for m in models:
            all_R.update(m.hop.keys())

        new_hop = dict()

        for R in all_R:
            hop_list = [np.array(m.hop[R]) for m in models]
            new_hop[R] = la.block_diag(*hop_list)

        return cls(dim=new_dim, uc=new_uc, pos=new_pos, occ=new_occ, hop=new_hop, contains_cc=False)

    def __add__(self, model: "Model") -> "Model":
        """
        Adds two models together by adding their hopping terms.
        """
        if not isinstance(model, Model):
            raise ValueError('Invalid argument type for Model.__add__: {}'.format(type(model)))

        # ---- CONSISTENCY CHECKS ----
        # check if the occupation number matches
        if self.occ != model.occ:
            raise ValueError(
                'Error when adding Models: occupation numbers ({0}, {1}) don\'t match'.format(self.occ, model.occ)
            )

        # check if the size of the hopping matrices match
        if self.size != model.size:
            raise ValueError(
                'Error when adding Models: the number of states ({0}, {1}) doesn\'t match'.format(
                    self.size, model.size
                )
            )

        # check if the unit cells match
        if not _check_compatibility.check_uc(self, model):
            raise ValueError(
                'Error when adding Models: unit cells don\'t match.\nModel 1:\n{0.uc}\n\nModel 2:\n{1.uc}'.format(
                    self, model
                )
            )

        # check if the positions match
        pos_match = True
        tolerance = 1e-6
        for v1, v2 in zip(self.pos, model.pos):
            if not pos_match:
                break
            for x1, x2 in zip(v1, v2):
                if abs(x1 - x2) > tolerance:
                    pos_match = False
                    break
        if not pos_match:
            raise ValueError(
                'Error when adding Models: positions don\'t match.\nModel 1:\n{0.pos}\n\nModel 2:\n{1.pos}'.format(
                    self, model
                )
            )

        # ---- MAIN PART ----
        new_hop = copy.deepcopy(self.hop)
        for R, hop_mat in model.hop.items():
            new_hop[R] += hop_mat
        # -------------------
        return Model(**co.ChainMap(dict(hop=new_hop), self._input_kwargs))

    def __sub__(self, model: "Model") -> "Model":
        """
        Substracts one model from another by substracting all hopping terms.
        """
        return self + -model

    def __neg__(self) -> "Model":
        """
        Changes the sign of all hopping terms.
        """
        return -1 * self

    def __mul__(self, x: float) -> "Model":
        """
        Multiplies hopping terms by x.
        """
        new_hop = dict()
        for R, hop_mat in self.hop.items():
            new_hop[R] = x * hop_mat

        return Model(**co.ChainMap(dict(hop=new_hop), self._input_kwargs))

    def __rmul__(self, x: float) -> "Model":
        """
        Multiplies hopping terms by x.
        """
        return self.__mul__(x)

    def __truediv__(self, x: float) -> "Model":
        """
        Divides hopping terms by x.
        """
        return self * (1. / x)
