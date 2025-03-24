Feature: Search Functionality Feature

@smoke @regression
Scenario: Search for a valid item
  Given the search bar is visible
  When the user enters "laptop" in the search bar
  And the user clicks the "search" button
  Then the search results should display items related to "laptop"
  And the results should include at least one item with "laptop" in the title

@smoke @regression
Scenario: Search for an invalid item
  Given the search bar is visible
  When the user enters "xyz123" in the search bar
  And the user clicks the "search" button
  Then the search results should display "No results found" message

@smoke @regression
Scenario: Search with an empty query
  Given the search bar is visible
  When the user leaves the search bar empty
  And the user clicks the "search" button
  Then the search results should display "Please enter a search term" message

@smoke @regression
Scenario: Search with special characters
  Given the search bar is visible
  When the user enters "!@#$%" in the search bar
  And the user clicks the "search" button
  Then the search results should display "No results found" message