import pandas as pd
from tqdm import tqdm
import torch
import numpy as np

from argparse import ArgumentParser, Namespace
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils.utils_new import load_pre_trained_model, create_corruption_function, load_target_image
from models.ddim import DDIMSampler
from BRGM_ddim_cond import project, sampling_from_ddim
from utils.plot_new import compare_cubes, plot_orthogonal_cuts
from utils.add_argument import add_argument
from utils.const import OUTPUT_FOLDER

def run_astro_test_suite(hparams: Namespace, test_files: list, device: torch.device):
    """
    Esegue il benchmark su un set di cubi FITS, calcola metriche 
    e salva plot comparativi e report CSV.
    """
    # 1. Inizializzazione modelli (una sola volta)
    diffusion, decoder = load_pre_trained_model(device=device)
    ddim = DDIMSampler(diffusion)
    forward = create_corruption_function(hparams=hparams, device=device)
    
    # Cartella di output per i risultati del test
    test_out_dir = Path(OUTPUT_FOLDER) / f"test_run_{hparams.experiment_name}"
    test_out_dir.mkdir(parents=True, exist_ok=True)
    
    results_data = []

    print(f"🚀 Inizio Test su {len(test_files)} oggetti astrofisici...")

    for i, file_path in enumerate(tqdm(test_files)):
        obj_name = Path(file_path).stem
        
        # A. Caricamento Ground Truth (GT)
        # Assicurati che load_target_image usi hparams.target_path
        hparams.target_path = str(file_path)
        img_tensor_gt = load_target_image(hparams, device=device) # [1, 1, D, H, W]
        
        # B. Inferenza (Inversione tramite ottimizzazione)
        # Creiamo un writer temporaneo per non sporcare quello principale
        temp_writer = SummaryWriter(log_dir=f"{hparams.tensor_board_logger}/test_{obj_name}")
        
        z_opt, cond_opt, _ = project(
            ddim, decoder, forward, img_tensor_gt, device, temp_writer, hparams, verbose=False
        )
        temp_writer.close()

        # C. Generazione Cubo Ricostruito Finale
        with torch.no_grad():
            restored_tensor = sampling_from_ddim(ddim, z_opt, decoder, cond_opt, hparams)
        
        # D. Preparazione dati per Plotting e Metriche
        
        synth_img_np = restored_tensor[0, 0].detach().cpu().numpy()
        target_np = img_tensor_gt[0, 0].detach().cpu().numpy()
        score_ssim = ssim(
            synth_img_np,
            target_np,
            win_size=11,
            data_range=1.0,
            gaussian_weights=True,
            use_sample_covariance=False,
        )
        # Code for computing PSNR is adapted from
        # https://github.com/agis85/multimodal_brain_synthesis/blob/master/error_metrics.py#L32
        data_range = np.max([synth_img_np.max(), target_np.max()]) - np.min(
            [synth_img_np.min(), target_np.min()]
        )
        score_psnr = psnr(target_np, synth_img_np, data_range=data_range)
        
        
        # E. Salvataggio Metriche nel Report
        results_data.append({
            "object_id": obj_name,
            "psnr": score_psnr,
            "ssim": score_ssim,
            "est_hi_size": cond_opt[0, 0].item(),
            "est_flux": cond_opt[0, 1].item(),
            "est_inclination": cond_opt[0, 2].item(),
            "est_w20": cond_opt[0, 3].item()
        })

        # F. Visualizzazione (Utilizzando le tue funzioni custom)
        # 1. Plot comparativo (Integrated Moment 0 + Residui)
        compare_cubes(
            gt_np, recon_np, 
            title=f"Obj: {obj_name} | PSNR: {score_psnr:.2f}",
            save_path=test_out_dir / f"{obj_name}_comparison.png"
        )
        
        # 2. Tagli Ortogonali (PV Diagrams) per il ricostruito
        plot_orthogonal_cuts(
            recon_np, 
            title=f"Reconstructed PV: {obj_name}", 
            ssim=score_ssim,
            save_path=test_out_dir / f"{obj_name}_pv_cuts.png"
        )

        # Pulizia memoria
        del img_tensor_gt, restored_tensor, gt_np, recon_np
        torch.cuda.empty_cache()

    # 2. Generazione Report Finale
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(test_out_dir / "test_benchmark_report.csv", index=False)
    
    # 3. Print riassuntivo
    print("\n✅ Test Benchmark Completato!")
    print("-" * 30)
    print(df_results[["psnr", "ssim"]].describe().loc[['mean', 'std', 'min', 'max']])
    
    return df_results

if __name__ == "__main__":
    parser = ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()
    
    # Crea una lista di file per il test
    test_path = Path("./data/test_npy/")
    test_files = list(test_path.glob("*.npy"))[:5] # Test sui primi 5
    
    device = torch.device(args.device)
    report_df = run_astro_test_suite(args, test_files, device)
