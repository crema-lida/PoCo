import io
import math
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image


def smiles_to_image(
    smiles: str,
    bond_px: float = 30,
    bond_width: float = 2,
    font_size: float = 16,
    atom_label_padding: float = 0.0,
    padding_px: int = 20,
    sanitize: bool = True,
    kekulize: bool = True,
    rotate_deg: float = 0.0,
    bg_color: tuple[int] | None = None,
    mirror: bool = False,
):
    """Render SMILES as a PIL image with a canvas sized to the molecule.

    Args:
        smiles: Molecule SMILES string.
        bond_px: Target bond length in pixels.
        bond_width: Bond line width in pixels.
        font_size: Atom label font size in pixels.
        atom_label_padding: Extra label padding as a fraction of the font size.
        padding_px: Padding per side when sizing the canvas, in pixels.
        sanitize: Sanitize the molecule when parsing SMILES.
        kekulize: Kekulize aromatic bonds for drawing.
        rotate_deg: Rotation about the molecular centroid, in degrees.
        bg_color: RGB or RGBA color with values in [0, 1]; None for transparency.
        mirror: Flip the molecule vertically before rotation.

    Returns:
        PIL.Image.Image: Rendered molecular structure.
    """

    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)

    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()

    if rotate_deg or mirror:
        xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
        ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        if mirror:
            for i in range(mol.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                conf.SetAtomPosition(i, (p.x, 2 * cy - p.y, p.z))
        if rotate_deg:
            theta = math.radians(float(rotate_deg))
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            for i in range(mol.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                x = p.x - cx
                y = p.y - cy
                xr = x * cos_t - y * sin_t + cx
                yr = x * sin_t + y * cos_t + cy
                conf.SetAtomPosition(i, (xr, yr, p.z))

    xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span_x = max_x - min_x
    span_y = max_y - min_y

    # Use the mean 2D bond length to estimate the coordinate-to-pixel scale.
    bond_lengths = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        pi = conf.GetAtomPosition(i)
        pj = conf.GetAtomPosition(j)
        dx = pi.x - pj.x
        dy = pi.y - pj.y
        bond_lengths.append((dx*dx + dy*dy) ** 0.5)

    if len(bond_lengths) == 0:
        avg_bond_len = 1.5
    else:
        avg_bond_len = sum(bond_lengths) / len(bond_lengths)

    scale = bond_px / avg_bond_len

    # Fit the canvas to the transformed molecular bounds, including padding.
    width_px  = int(math.ceil(span_x * scale + 2 * padding_px))
    height_px = int(math.ceil(span_y * scale + 2 * padding_px))

    # Use the drawer API to set bond length and font size explicitly.
    drawer = rdMolDraw2D.MolDraw2DCairo(width_px, height_px)
    dopts = drawer.drawOptions()

    dopts.fixedBondLength = bond_px
    dopts.bondLineWidth = bond_width
    dopts.fixedFontSize = font_size
    dopts.additionalAtomLabelPadding = atom_label_padding
    dopts.useBWAtomPalette()
    if bg_color is not None:
        dopts.setBackgroundColour(bg_color)
    else:
        dopts.setBackgroundColour((1, 1, 1, 0))

    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, kekulize=kekulize)
    drawer.FinishDrawing()

    png = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(png))
    return img
