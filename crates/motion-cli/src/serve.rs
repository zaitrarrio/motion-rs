#[derive(clap::Args, Debug)]
pub struct ServeArgs {}
pub fn run<B: motion_core::backend::Backend>(_: B, _: ServeArgs) -> anyhow::Result<()> {
    Ok(())
}
