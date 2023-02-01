const path = require("path");
const fs = require("fs");
const cracoBabelLoader = require("craco-babel-loader");

// manage relative paths to packages
const appDirectory = fs.realpathSync(process.cwd());
const resolvePackage = relativePath => path.resolve(appDirectory, relativePath);

module.exports = {
  devServer: {
    // When launching `yarn start dev`, write the files to the build folder too
    devMiddleware: { writeToDisk: true },
  },
  webpack: {
    configure: {
      output: {
        publicPath: "./",
      },
    },
  },
  plugins: [
    {
      plugin: cracoBabelLoader,
      options: {
        includes: [resolvePackage("node_modules/lightning-ui")],
      },
    },
  ],
};
