## Description 
Currently in progress, this project is meant to provide a website interface using Shiny for Python, allowing users to first identify their utility provider by zip code, and then based on that information see information about the origins and sources of their energy within a specified data range. 

## Notes for self
sources:
- EIA website grid monitor
https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/US48/US48 


- Ce5.3 data for average household energy consumption by state
https://www.eia.gov/consumption/residential/data/2020/index.php?view=state#ce 
--> https://www.eia.gov/consumption/residential/data/2020/state/pdf/ce5.3.st.pdf

- has more columns/appliances but is sorted by U.S. Regions
https://www.eia.gov/consumption/residential/data/2020/c&e/pdf/ce5.3b.pdf 

- zip codes (2023)
https://catalog.data.gov/dataset/u-s-electric-utility-companies-and-rates-look-up-by-zipcode-2023/resource/8c63c38b-4d95-436b-9baf-83e650058571

- EIA-861 Form which helps match balancing authority to eiaid
https://www.eia.gov/electricity/data/eia861/ 


Additional Sources
- show monthly weather history, proves PowerSource Trends in app graph specifically when looking at March solar.
https://www.wunderground.com/calendar/us/az/phoenix/KPHX/date/2025-3
- Example of utility provider website that did not work:
https://www.arcgis.com/apps/InformationLookup/index.html?appid=db6e5122fa8245f78f3b4d7b3f474c62

ideas: 
- navbar for home page, and have zipcode box in navbar, once users enter zip code, they can navigate between different pages while giving each page access to the zip code.
For example, have the PowerSources page use zip code to identify utility provider and provide PowerSource, PowerDemand insights. 
Then have HomeEnergyConsumption page use zip code to identify state, and provide average usage statistics on appliances. 
- Use replace current input validation methods by using shiny-validate module
