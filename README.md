# Science Diplomacy: A Global Research Field?

This repository contains the scripts, data, and results associated with the study:

Rüland, AL., Andersen, L.H., Hassen, A., Kinyanjui, C., Ralfs, A., Grisci, B.I. _Science diplomacy: A global research field? Findings from a bibliometric analysis of the science diplomacy scholarship of the past twenty years._ **Scientometrics** (2025). [https://doi.org/10.1007/s11192-025-05396-x](https://doi.org/10.1007/s11192-025-05396-x)

## 🧾 Abstract

Science diplomacy is a unique and evolving field shaped by both academic and policy actors. This study explores how recent developments in the discourse around science diplomacy are reflected in the academic literature. Specifically, it examines:

- The internationalization of the field through the geographical distribution of authors, funding sources, and research focus.
- The extent to which calls for greater diversity in science diplomacy scholarship have been addressed.

Using **network analysis** and **large language model-enhanced bibliometric techniques**, the results indicate a slow and uneven internationalization of the field, with the **United States and Europe** remaining dominant. The findings reveal persisting **North-South dynamics**, highlighting the need for broader global inclusion in science diplomacy research.

## 📁 Repository Structure

```text
├── data/
│   ├── akh_v5_1209_final_worksheet.xlsx
│   │   └── Primary dataset used for bibliometric analysis
│   └── 22072024_Countries aggregated_BG-3007 - UN Geoscheme.csv
│       └── Country groupings based on UN Geoscheme
│
├── networks/
│   ├── .gexf, .pdf, .html files for all the networks
│   └── Centrality measures for nodes in the author and region networks
│
├── scripts/
│   └── *.py
│       └── Python scripts for data processing, network construction, and visualization
│
└── README.md
```
## 📄 Key Dataset Files

- 📊 [akh_v5_1209_final_worksheet.xlsx](./akh_v5_1209_final_worksheet.xlsx): Main dataset used for the bibliometric review and network construction (Excel)
- 📊 [akh_v5_1209_final_worksheet.csv](./akh_v5_1209_final_worksheet.csv): Main dataset used for the bibliometric review and network construction (CSV)
- 🌍 [22072024_Countries aggregated_BG-3007 - UN Geoscheme.csv](<./22072024_Countries aggregated_BG-3007 - UN Geoscheme.csv>): Country-to-region mapping based on the UN Geoscheme

## 🌐 Interactive Networks

Interactive versions of the collaboration networks can be accessed through the links below:

- **Author Collaboration Network**  
  [https://brunogrisci.github.io/scidip/authornet_2002_2023.html](https://brunogrisci.github.io/scidip/authornet_2002_2023.html)

- **Region Targeted Network**  
  [https://brunogrisci.github.io/scidip/regionnet_2002_2023.html](https://brunogrisci.github.io/scidip/regionnet_2002_2023.html)

These interactive visualizations allow users to explore nodes, relationships, and centrality values dynamically in their web browsers. They complement the static PDF and GEXF network files stored in the `networks/` folder.

## 📚 Citation

If you use this repository, the data, or visualizations in your own work, please cite the following study:

Rüland, AL., Andersen, L.H., Hassen, A., Kinyanjui, C., Ralfs, A., Grisci, B.I. _Science diplomacy: A global research field? Findings from a bibliometric analysis of the science diplomacy scholarship of the past twenty years._ **Scientometrics** (2025). [https://doi.org/10.1007/s11192-025-05396-x](https://doi.org/10.1007/s11192-025-05396-x)


## 📬 Contact

**Anna-Lena Rüland**: a.ruland@ucl.ac.uk

**Bruno Iochins Grisci**: bigrisci@inf.ufrgs.br