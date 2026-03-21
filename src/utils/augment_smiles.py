import random
import re
from types import NoneType
from psmiles import PolymerSmiles as PS
from rdkit import Chem
from rdkit.Chem import rdchem

from .utils import parallel_apply

BRACKET_STAR = re.compile(r"(\[\*\])") 


def canonicalize(s):
    return PS(s).canonicalize.psmiles


def parallel_canonicalize(smiles_list):
    return parallel_apply(smiles_list, canonicalize)


def replace_stars(s: str) -> str:
    parts = BRACKET_STAR.split(s)
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace("*", "[*]")
    return "".join(parts)


def translate_smiles(smiles: str, position: int | None = None) -> str:
    """Shift the repeat-unit window of `smiles` to a new position.

    Only backbone bonds that are **not** part of any ring may be broken.
    """
    ps = PS(smiles)
    info = ps.get_connection_info()
    path = info["neighbor"]["path"] or ()

    if len(path) < 3:  # need at least two bonds to shift
        return smiles

    mol = ps.mol
    ring_bonds = set()
    for bond_idx in mol.GetRingInfo().BondRings():
        ring_bonds.update(bond_idx)

    # Find indices of two connected atoms that can be broken
    candidates = []
    for ai, aj in zip(path[:-1], path[1:]):
        bond = mol.GetBondBetweenAtoms(ai, aj)
        if bond.GetIdx() in ring_bonds:
            continue
        candidates.append((ai, aj))

    if not candidates:
        return smiles

    # Choose a random atom pair if none given
    if position is None:
        ai, aj = random.choice(candidates)
    else:
        ai, aj = candidates[position % len(candidates)]
    
    # Create periodic ring closing the two stars
    bond_type = info["neighbor"]["bond_type"][0][0]
    mol.AddBond(path[0], path[-1], bond_type)

    # Remove old stars
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    star_idx = [i for i, sym in enumerate(symbols) if sym == "*"]
    for i in sorted(star_idx, reverse=True):
        mol.RemoveAtom(i)
    
    # Adjust indices after removal
    ai = ai - (1 if ai > star_idx[0] else 0)
    aj = aj - (1 if aj > star_idx[0] else 0)

    # Fragment on selected bond, introduce new stars
    mol = Chem.FragmentOnBonds(
        mol, [mol.GetBondBetweenAtoms(ai, aj).GetIdx()], addDummies=True, dummyLabels=[(0, 0)]
    )
    sm = Chem.MolToSmiles(mol)
    return replace_stars(sm)


def _get_star_info(mol: Chem.Mol):
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "*"]
    if len(stars) != 2:
        raise ValueError(f"PSMILES 必须且只能有两颗星号，当前有 {len(stars)}")
    info = []
    for s in stars:
        nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(s).GetNeighbors()]
        if len(nbrs) != 1:
            raise ValueError("每个星号原子应只有一个邻居（端点）。")
        b = mol.GetBondBetweenAtoms(s, nbrs[0]).GetBondType()
        info.append((s, nbrs[0], b))
    return info  # [(star_idx, neighbor_idx, bond_type), (..)]


def multiply_smiles(psmiles: str,
                    n: int = 2,
                    sanitize: bool = True,
                    strict_bond: bool = True) -> str:
    """
    将带两端点的 PSMILES 重复 N 次并线性连接，保留最外侧两端的 [*]。

    参数
    ----
    psmiles        : 输入（允许 * 或 [*]；允许端点为配位键 <- 或 ->）
    n              : 复制次数 (>=1)
    bond_mode      : 'inherit'  继承端点键型（SINGLE/DOUBLE/TRIPLE/DATIVE）
                     'single'   聚合键一律用单键（推荐处理端点是配位键的场景）
    sanitize_mode  : 'strict' | 'loose' | 'auto'（见 _sanitize 说明）
    strict_bond    : True 时要求左右端点键型一致；否则将按 bond_mode 处理
    """
    if n < 1:
        raise ValueError("n 必须 >= 1")

    base = Chem.MolFromSmiles(psmiles, sanitize=False)  # 读入时先不 sanitize，兼容奇异端点
    if base is None:
        raise ValueError(f"无效的 SMILES: {psmiles}")

    # 端点与键阶
    (sL, nL, bL), (sR, nR, bR) = _get_star_info(base)
    if strict_bond and (bL != bR):
        raise ValueError(f"两端点的键型不一致：左 {bL} vs 右 {bR}")
    # 端点是配位键时，很多体系（如 Si/ Sn）严格价态会失败；可考虑改成单键
    connect_order = bL if (bL == bR) else rdchem.BondType.SINGLE
    if connect_order == rdchem.BondType.DATIVE:
        # 默认把“端点配位键”的聚合连接降级为单键，避免价态炸掉
        connect_order = rdchem.BondType.SINGLE

    if n == 1:
        sm = Chem.MolToSmiles(base, canonical=True)
        sm = re.sub(r'(?<!\[)\*(?!\])', '[*]', sm)
        return sm

    # 合并 N 份
    combo = base
    for _ in range(n - 1):
        combo = Chem.CombineMols(combo, base)
    rw = Chem.RWMol(combo)
    na = base.GetNumAtoms()

    def idx_left(i):
        return sL + i*na, nL + i*na
    def idx_right(i):
        return sR + i*na, nR + i*na

    # 右邻居(i) —— 左邻居(i+1) 直接成键
    for i in range(n - 1):
        _, nR_i = idx_right(i)
        _, nL_j = idx_left(i + 1)
        rw.AddBond(nR_i, nL_j, connect_order)

    # 删除内部星号（只保留最外侧两端）
    stars_to_delete = []
    for i in range(n):
        sL_i, _ = idx_left(i)
        sR_i, _ = idx_right(i)
        if i >= 1:      stars_to_delete.append(sL_i)
        if i <= n - 2:  stars_to_delete.append(sR_i)
    for aidx in sorted(stars_to_delete, reverse=True):
        rw.RemoveAtom(aidx)

    mol = rw.GetMol()
    if sanitize:
        Chem.SanitizeMol(mol)
    sm = Chem.MolToSmiles(mol, canonical=True)
    sm = re.sub(r'(?<!\[)\*(?!\])', '[*]', sm)

    # 用“图结构”确认端点个数
    num_stars = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "*")
    if num_stars != 2:
        raise ValueError(f"结果端点不是两处（剩 {num_stars} 处）：{sm}")

    return sm


def randomize_smiles(smiles: str) -> str:
    sm = Chem.MolToSmiles(
        Chem.MolFromSmiles(smiles), doRandom=True,
    )
    return replace_stars(sm)


def augment_smiles(
    base: str | None = None,
    anchor: str | list[str] | None = None,
    translate: bool = True,
    multiply: bool = True,
    permute: bool = True,
    max_length: int = 200,
    max_mult: int = 1,
    max_tries: int = 10,
) -> str:
    """Returns a new SMILES string *different* from `anchor` by:
       • Translation: Move the repeat unit window.   *CCO* -> *COC*
       • Multiplication: Make a twofold repeat unit.   *C* -> *CC*
       • Permutation: Do syntactical permutations.     *C* -> C(*)*
    
    If `base` is provided, it will be used as the original SMILES.
    Otherwise, a random SMILES from `anchor` will be used.
    Eligible augmentations are applied sequentially.
    """

    assert anchor or base, "Either `anchor` or `base` must be provided."
    anchor = [anchor] if isinstance(anchor, (str, NoneType)) else anchor
    sm_aug = random.choice(anchor) if base is None else base

    if translate:
        sm_aug = translate_smiles(sm_aug)
    
    if multiply:
        sm_length = len(sm_aug)
        _max_mult = min(max_mult, max_length // sm_length - 1)
        _max_mult = max(_max_mult, 0)
        n_mult = random.randint(0, _max_mult)
        if n_mult > 0:
            sm_aug = multiply_smiles(sm_aug, n_mult + 1)

    if permute:
        sm_aug = randomize_smiles(sm_aug)    

    for _ in range(max_tries):
        if sm_aug not in anchor:
            return sm_aug
        sm_aug = augment_smiles(base, anchor, translate, multiply, permute,
                                max_length, max_mult, max_tries=0)
    return sm_aug
