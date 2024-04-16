using SchlierenReconstructions
using Documenter

DocMeta.setdocmeta!(SchlierenReconstructions, :DocTestSetup, :(using SchlierenReconstructions); recursive=true)

makedocs(;
    modules=[SchlierenReconstructions],
    authors="Masahito Akamine <akamine502@gmail.com>",
    sitename="SchlierenReconstructions.jl",
    format=Documenter.HTML(;
        canonical="https://wavepackets.github.io/SchlierenReconstructions.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/wavepackets/SchlierenReconstructions.jl",
    devbranch="main",
)
