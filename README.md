## Description 
Currently in progress, this project is meant to provide a website interface using Shiny for Python, allowing users to first identify their utility provider by zip code, and then based on that information see information about the origins and sources of their energy within a specified data range. 

## Notes for self
sources:
- data link
https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/US48/US48 

- https://www.eia.gov/consumption/residential/data/2020/index.php?view=state#ce 
--> https://www.eia.gov/consumption/residential/data/2020/state/pdf/ce5.3.st.pdf

- has more columns/appliances but is sorted by U.S. Regions
https://www.eia.gov/consumption/residential/data/2020/c&e/pdf/ce5.3b.pdf 



Example of utility provider website that did not work:
https://www.arcgis.com/apps/InformationLookup/index.html?appid=db6e5122fa8245f78f3b4d7b3f474c62

ideas: 
add navbar for home page, and have zipcode box in navbar, once users enter zip code, they can navigate between different pages while giving each page access to the zip code.
For example, have the PowerSources page use zip code to identify utility provider and provide PowerSource, PowerDemand insights. 
Then have HomeEnergyConsumption page use zip code to identify state, and provide average usage statistics on appliances. 
