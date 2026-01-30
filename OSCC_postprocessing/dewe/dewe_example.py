from dewe import * 

# file_path = r"F:\G\MeOH_test\Dewe\T2_0001.dxd"
file_path = r"F:\G\MeOH_test\Dewe\T56_0001.dxd"

def main() -> None:
    path = resolve_data_path(file_path)
    df = load_dataframe(path)

    # First pass: create independent figures/axes, don't show yet
    _fig1, _ax1 = cast(Tuple[Figure, Axes], plot_dataframe(
        df, title=path.name, 
        criteria=["Chamber Pressure", "Chamber gas temperature"],
        return_fig=True, show=False
    ))

    df_heat_release = df[[col for col in df.columns if "Heat Release" in col]].copy()
    if "Heat Release" in df_heat_release.columns:
        df_heat_release["Heat Release"] = df_heat_release["Heat Release"] * 100  # scale for visibility

    # plot_dataframe(df, criteria=["Heat Release"], ax=_ax1, show=False)
    _fig2, _ax2 = cast(Tuple[Figure, Axes], plot_dataframe(
        df_heat_release, title=path.name, 
        criteria=["Heat Release"],
        return_fig=True, show=False
    ))


    _fig3, _ax3 = cast(Tuple[Figure, Axes], plot_dataframe(
        df, title=path.name, 
        criteria=["Current Profile"],
        return_fig=True, show=False
    ))

    from heat_release_calulation import hrr_calc
    chamber_pressure_col = next((c for c in df.columns if "chamber pressure" in c.lower()), None)
    if chamber_pressure_col is None:
        raise RuntimeError("No chamber pressure column found for HRR calculation")
    ChmbP_bar = df[chamber_pressure_col].to_numpy()
    time_seconds = df.index.to_numpy() if df.index.name == "time_s" else None
    df_hrr = hrr_calc(ChmbP_bar, time=time_seconds, V_m3=8.5e-3, gamma=1.35)
    # First plot (pressure-derived HRR)
    fig4, ax4 = plot_dataframe(
        df_hrr.set_index("time_s").rename(columns={"HRR_W": "Heat Release Rate"}),
        title="HRR",
        criteria=["Heat Release Rate"], return_fig=True
    )
    plt.show()



if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc)
        sys.exit()
